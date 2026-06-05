"""Per-night photometric calibration post-stage.

Consumes the manifest + per-batch ``SenpaiRun`` JSONs that
:mod:`senpai.cli.burr` writes, aggregates each frame's
:class:`~senpai.engine.photometry.utils.SimplePhotometrySummary` into a
:class:`NightCalibration`, and emits a calibration JSON + plot set:

* zero point vs airmass (Bouguer extinction fit per filter),
* limiting magnitude distribution per filter,
* Az/Alt coverage polar plot (one point per frame),
* ZP drift over the night.

The shape of this module is deliberately narrow: no FITS I/O, no astrometry,
no photometry. Frames without WCS still contribute to per-filter ZP medians;
geometric aggregates (extinction, Az/Alt coverage) drop those frames.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- per-frame extract --------------------------------------------------------


@dataclass(slots=True)
class FramePhoto:
    """Calibration-relevant slice of one frame's senpai output."""

    batch_id: str
    frame_index: int
    timestamp: datetime | None
    track_mode: str | None  # 'sidereal' or 'rate'
    filter_name: str | None
    exposure_time: float | None
    zero_point: float | None
    zero_point_err: float | None
    limiting_magnitude_50: float | None
    limiting_magnitude_90: float | None
    median_snr: float | None
    median_background: float | None
    n_stars: int | None
    n_quality: int | None

    # Geometry (None if WCS didn't solve).
    ra_center_deg: float | None = None
    dec_center_deg: float | None = None
    altitude_deg: float | None = None
    azimuth_deg: float | None = None
    airmass: float | None = None
    fov_sq_deg: float | None = None  # x_fov × y_fov from WCS, for area metrics

    # Filter → ZP from the multiband calibration, when present.
    multiband_zps: dict[str, float] = field(default_factory=dict)

    # Per-star catalog magnitude + measured SNR pairs, lifted from the
    # SimplePhotometrySummary that senpai's collect pipeline now retains.
    # Empty (not None) when the frame has photometry but no qualifying stars.
    stars_mag: list[float] = field(default_factory=list)
    stars_snr: list[float] = field(default_factory=list)
    # Per-star ZP offset (m_cat − m_inst), parallel to stars_mag; None entries
    # mark stars without a measured instrumental magnitude. Empty on runs
    # predating stars_zp_offset retention.
    stars_zp_offset: list[float | None] = field(default_factory=list)
    # Per-star isolation flag, parallel to stars_mag (False = a brighter
    # catalog star inside the aperture footprint — blended flux). Empty on
    # runs predating retention; treat missing as isolated.
    stars_isolated: list[bool] = field(default_factory=list)

    # Rate-track geometry (None on sidereal frames). Retained for *all* frames
    # — including rate — so limiting-case studies are possible: ZP(streak
    # length), ZP(track rate), ΔSNR vs magnitude & streak length, and where
    # rate frames run out of stars (fast rates / small FoV). The night-level ZP
    # aggregation excludes rate frames (see _zp_frames), but the per-frame raw
    # numbers here are kept regardless.
    streak_length_px: float | None = None
    streak_fwhm_px: float | None = None
    pixel_track_rate: float | None = None  # px/s
    track_rate_arcsec_per_s: float | None = None  # on-sky |rate|, plate-scale independent

    @property
    def has_wcs(self) -> bool:
        return self.ra_center_deg is not None and self.dec_center_deg is not None


def _safe_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _airmass(altitude_deg: float | None) -> float | None:
    """Plane-parallel airmass = sec(zenith). Good to ~3.5; we clip above
    altitude < 3° to avoid divergence in plots."""
    if altitude_deg is None or altitude_deg <= 3.0:
        return None
    z = math.radians(90.0 - altitude_deg)
    return 1.0 / math.cos(z)


def _compute_alt_az(
    ra_deg: float, dec_deg: float, when: datetime, site: dict[str, Any] | None
) -> tuple[float, float] | None:
    """RA/Dec + UTC + site → (altitude_deg, azimuth_deg). Returns None if astropy
    is unavailable or site is incomplete. Imports are lazy so a calibration
    can be loaded without astropy when only ZP aggregates are needed."""

    if not site or site.get("latitude") is None or site.get("longitude") is None:
        return None

    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        from astropy import units as u
    except ImportError:
        logger.debug("astropy unavailable; skipping alt/az conversion")
        return None

    location = EarthLocation(
        lat=site["latitude"] * u.deg,
        lon=site["longitude"] * u.deg,
        height=(site.get("altitude_km") or 0.0) * 1000.0 * u.m,
    )
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    aa = sky.transform_to(AltAz(obstime=Time(when), location=location))
    return float(aa.alt.deg), float(aa.az.deg)


def _extract_frame_photo(
    frame_dict: dict[str, Any],
    batch_id: str,
    site: dict[str, Any] | None,
    track_mode_default: str,
) -> FramePhoto | None:
    """Pull a FramePhoto from one serialized ``SiderealFrameSerializable`` or
    ``RateTrackFrameSerializable``. Returns None for frames with no photometry
    summary at all — they carry no calibration signal."""

    summary = frame_dict.get("photometry_summary")
    if not summary:
        # No photometry was measured; nothing to aggregate.
        return None

    fmd = frame_dict.get("frame_metadata") or {}
    starfield = frame_dict.get("starfield") or {}
    wcs_meta = (starfield or {}).get("wcs_metadata") or {}

    ts = _safe_iso(frame_dict.get("timestamp") or fmd.get("observation_time"))

    # Multiband ZPs, when present, are nested under photometry_summary.
    mb_cal = summary.get("multiband_calibration") or {}
    multiband_zps: dict[str, float] = {}
    for band_name, band in (mb_cal.get("bands") or {}).items():
        zp = band.get("zero_point") if isinstance(band, dict) else None
        if zp is not None:
            multiband_zps[band_name] = float(zp)

    ra_center = wcs_meta.get("RA_center_deg")
    dec_center = wcs_meta.get("Dec_center_deg")

    fov_x = wcs_meta.get("x_fov_degrees")
    fov_y = wcs_meta.get("y_fov_degrees")
    fov_sq_deg = float(fov_x) * float(fov_y) if fov_x and fov_y else None

    altitude = azimuth = None
    if ra_center is not None and dec_center is not None and ts is not None:
        alt_az = _compute_alt_az(ra_center, dec_center, ts, site)
        if alt_az is not None:
            altitude, azimuth = alt_az

    track_mode = fmd.get("track_mode") or track_mode_default

    # Rate-track geometry — only meaningful on rate frames. (Burr leaves a
    # non-zero residual RA/DEC rate in the header even on the sidereal leg, so
    # we must not read a "track rate" off sidereal frames.)
    streak_length_px = streak_fwhm_px = pixel_track_rate = None
    track_rate_arcsec_per_s = None
    if track_mode == "rate":
        streak = frame_dict.get("streak") or {}
        streak_length_px = streak.get("pixel_length")
        streak_fwhm_px = streak.get("fwhm")
        pixel_track_rate = frame_dict.get("pixel_track_rate_per_second")
        rate_ra = fmd.get("track_rate_ra_arcsec_per_second")
        rate_dec = fmd.get("track_rate_dec_arcsec_per_second")
        if rate_ra is not None and rate_dec is not None:
            track_rate_arcsec_per_s = math.hypot(rate_ra, rate_dec)

    return FramePhoto(
        batch_id=batch_id,
        frame_index=int(frame_dict.get("index", -1)),
        timestamp=ts,
        track_mode=track_mode,
        filter_name=fmd.get("observation_filter"),
        exposure_time=fmd.get("exposure_time_seconds"),
        zero_point=summary.get("zero_point"),
        zero_point_err=summary.get("zero_point_err"),
        limiting_magnitude_50=summary.get("limiting_magnitude_50"),
        limiting_magnitude_90=summary.get("limiting_magnitude_90"),
        median_snr=summary.get("median_snr"),
        median_background=summary.get("median_background"),
        n_stars=summary.get("n_stars"),
        n_quality=summary.get("n_quality"),
        ra_center_deg=ra_center,
        dec_center_deg=dec_center,
        altitude_deg=altitude,
        azimuth_deg=azimuth,
        airmass=_airmass(altitude),
        fov_sq_deg=fov_sq_deg,
        multiband_zps=multiband_zps,
        stars_mag=list(summary.get("stars_mag") or []),
        stars_snr=list(summary.get("stars_snr") or []),
        stars_zp_offset=list(summary.get("stars_zp_offset") or []),
        stars_isolated=list(summary.get("stars_isolated") or []),
        streak_length_px=streak_length_px,
        streak_fwhm_px=streak_fwhm_px,
        pixel_track_rate=pixel_track_rate,
        track_rate_arcsec_per_s=track_rate_arcsec_per_s,
    )


# --- aggregates ---------------------------------------------------------------


@dataclass(slots=True)
class ZeroPointStat:
    """Per-filter summary statistic of zero points across a night."""

    filter_name: str
    n: int
    median: float
    p16: float  # 16th percentile (lower 1-σ-ish)
    p84: float  # 84th percentile (upper 1-σ-ish)
    median_err: float | None = None


@dataclass(slots=True)
class ExtinctionFit:
    """Bouguer linear fit ``zero_point = m0 - k * airmass`` over a filter's
    frames. ``k`` is the extinction coefficient (mag/airmass, conventionally
    positive on clear nights); ``m0`` is the extra-atmospheric zero point."""

    filter_name: str
    m0: float          # zero point at zero airmass (extra-atmospheric)
    m0_err: float
    k: float           # extinction (mag/airmass) — positive on clear nights
    k_err: float
    n: int
    airmass_range: tuple[float, float]


@dataclass(slots=True)
class NightCalibration:
    """Aggregated calibration products for one night."""

    night_id: str
    sensor: str | None
    site: dict[str, Any] | None
    n_frames_total: int
    n_frames_with_photometry: int
    n_frames_with_wcs: int
    frames: list[FramePhoto] = field(default_factory=list)
    zp_per_filter: dict[str, ZeroPointStat] = field(default_factory=dict)
    extinction_per_filter: dict[str, ExtinctionFit] = field(default_factory=dict)
    limiting_mag_p50_per_filter: dict[str, float] = field(default_factory=dict)
    limiting_mag_p90_per_filter: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "night_id": self.night_id,
            "sensor": self.sensor,
            "site": self.site,
            "n_frames_total": self.n_frames_total,
            "n_frames_with_photometry": self.n_frames_with_photometry,
            "n_frames_with_wcs": self.n_frames_with_wcs,
            "zp_per_filter": {
                k: _asdict_safe(v) for k, v in self.zp_per_filter.items()
            },
            "extinction_per_filter": {
                k: _asdict_safe(v) for k, v in self.extinction_per_filter.items()
            },
            "limiting_mag_p50_per_filter": self.limiting_mag_p50_per_filter,
            "limiting_mag_p90_per_filter": self.limiting_mag_p90_per_filter,
            "frames": [_asdict_safe(f) for f in self.frames],
        }


def _asdict_safe(obj: Any) -> Any:
    """Pure-stdlib dataclass→dict that handles our datetime fields."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        d = asdict(obj)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d
    return obj


# --- aggregation engine -------------------------------------------------------


# A Bouguer fit needs the airmass to actually vary; below this span the slope
# (extinction coefficient) is unconstrained and any fit is noise.
_MIN_AIRMASS_RANGE = 0.15


def _zp_frames(frames: list[FramePhoto]) -> list[FramePhoto]:
    """Frames whose photometry is trustworthy for the night's zero point: the
    sidereal-tracked frames. Rate-tracked frames image stars as streaks, so
    their aperture photometry — and any ZP derived from it — is unreliable and
    must not define the night's photometric calibration."""

    return [f for f in frames if f.track_mode == "sidereal"]


def _isolated_flags(f: FramePhoto) -> list[bool]:
    """Per-star isolation flags parallel to ``stars_mag``; legacy runs that
    predate ``stars_isolated`` retention default to all-isolated."""

    return f.stars_isolated or [True] * len(f.stars_mag)


def _snr_consistent(snr: float, mag: float, f: FramePhoto,
                    tolerance: float = 5.0) -> bool:
    """True when a star's measured SNR is plausible for its catalog magnitude.

    By definition SNR ≈ limiting_snr (3) at the frame's lim50 and scales with
    flux (×10^0.4 per mag) in the background-dominated regime, so the frame
    predicts each star's SNR from its magnitude alone. Stars measured far
    above that (×{tolerance}) are real flux wrongly attributed — bright-star
    wings/spikes outside the isolation radius, variables, bad cross-matches.
    They are ~2% of isolated SNR≥5 stars but carry ×10–80 flux excess, enough
    to fabricate entire faint-end bins. Saturated/bright stars always pass
    (measured ≤ predicted there)."""

    if f.limiting_magnitude_50 is None:
        return True
    snr_pred = 3.0 * 10 ** (0.4 * (f.limiting_magnitude_50 - mag))
    return snr <= tolerance * snr_pred


def _summarize_zp(frames: list[FramePhoto]) -> dict[str, ZeroPointStat]:
    """Median + 16/84 percentile of zero_point per filter (sidereal frames)."""

    import statistics

    by_filter: dict[str, list[FramePhoto]] = {}
    for f in _zp_frames(frames):
        if f.zero_point is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append(f)

    out: dict[str, ZeroPointStat] = {}
    for filt, fs in by_filter.items():
        zps = sorted(f.zero_point for f in fs)
        errs = [f.zero_point_err for f in fs if f.zero_point_err is not None]
        out[filt] = ZeroPointStat(
            filter_name=filt,
            n=len(zps),
            median=statistics.median(zps),
            p16=_percentile(zps, 0.16),
            p84=_percentile(zps, 0.84),
            median_err=statistics.median(errs) if errs else None,
        )
    return out


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (idx - lo) * (sorted_xs[hi] - sorted_xs[lo])


def _fit_extinction(frames: list[FramePhoto]) -> dict[str, ExtinctionFit]:
    """Per-filter Bouguer fit ``zero_point = m0 - k * airmass``.

    Requires ≥3 frames with both ZP and airmass in a filter. Standard OLS on
    the line ``y = m0 + slope * x``; the extinction coefficient ``k`` is the
    negated slope so it matches the conventional positive-extinction sign
    (atmosphere makes stars dimmer at higher airmass → ZP decreases with
    airmass → slope < 0 → k = -slope > 0).
    """

    by_filter: dict[str, list[tuple[float, float]]] = {}
    for f in _zp_frames(frames):
        if f.zero_point is None or f.airmass is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append((f.airmass, f.zero_point))

    out: dict[str, ExtinctionFit] = {}
    for filt, pairs in by_filter.items():
        if len(pairs) < 3:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        # Skip degenerate fits where airmass barely varies (e.g. a single
        # pointing) — the slope would be wild extrapolation.
        if max(xs) - min(xs) < _MIN_AIRMASS_RANGE:
            continue
        n = len(pairs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        ssxx = sum((x - mean_x) ** 2 for x in xs)
        ssxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        if ssxx <= 0:
            continue
        slope = ssxy / ssxx
        m0 = mean_y - slope * mean_x
        # Residual stderr against the actual fit line (slope, m0).
        resid = [ys[i] - (m0 + slope * xs[i]) for i in range(n)]
        ss_res = sum(r * r for r in resid)
        sigma2 = ss_res / max(n - 2, 1)
        slope_err = math.sqrt(sigma2 / ssxx) if sigma2 > 0 else 0.0
        m0_err = math.sqrt(sigma2 * (1 / n + mean_x * mean_x / ssxx)) if sigma2 > 0 else 0.0
        out[filt] = ExtinctionFit(
            filter_name=filt,
            m0=m0, m0_err=m0_err,
            k=-slope, k_err=slope_err,  # extinction coefficient is -slope
            n=n,
            airmass_range=(min(xs), max(xs)),
        )
    return out


def _summarize_limiting_mag(
    frames: list[FramePhoto], attr: str
) -> dict[str, float]:
    """Median limiting magnitude per filter, from SIDEREAL frames only.

    Rate-tracked frames image stars as streaks; their forced-photometry
    completeness is dominated by faint catalog positions landing on brighter
    stars' trails (a spurious detection floor), so a limiting mag read off them
    is unreliable. The night's authoritative depth comes from the sidereal legs
    (the raw rate per-frame values are still retained on each FramePhoto for
    limiting-case studies)."""
    import statistics

    by_filter: dict[str, list[float]] = {}
    for f in _zp_frames(frames):
        v = getattr(f, attr)
        if v is None:
            continue
        key = f.filter_name or "unknown"
        by_filter.setdefault(key, []).append(v)
    return {k: statistics.median(v) for k, v in by_filter.items()}


# --- night loader -------------------------------------------------------------


def analyze_night(night_dir: str | Path) -> NightCalibration:
    """Build a :class:`NightCalibration` from the output of
    ``python -m senpai.cli.burr night <night_dir> -o <output>`` — i.e. a dir
    that contains ``manifest.json`` and a ``batches/`` tree of SenpaiRun JSONs."""

    night_dir = Path(night_dir)
    manifest_path = night_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json at {manifest_path}. Run `senpai-burr night` first."
        )
    manifest = json.loads(manifest_path.read_text())

    frames: list[FramePhoto] = []
    n_total = 0
    for entry in manifest.get("batches", []):
        # A skipped batch (resumed --skip-existing run) still has valid output;
        # include it as long as its result JSON is present.
        result_path = entry.get("result_path")
        if not result_path:
            continue
        batch_id = entry["batch_id"]
        path = Path(result_path)
        if not path.is_file():
            # The manifest stores absolute paths; when the drive remounts
            # elsewhere they go stale. Re-anchor on this night_dir.
            path = night_dir / "batches" / batch_id / path.name
            if not path.is_file():
                continue
        try:
            run = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            logger.warning("Skipping unreadable %s: %s", path, e)
            continue
        for fd in run.get("sidereal_frames", []):
            n_total += 1
            fp = _extract_frame_photo(fd, batch_id, manifest.get("site"), "sidereal")
            if fp is not None:
                frames.append(fp)
        for fd in run.get("rate_track_frames", []):
            n_total += 1
            fp = _extract_frame_photo(fd, batch_id, manifest.get("site"), "rate")
            if fp is not None:
                frames.append(fp)

    n_with_wcs = sum(1 for f in frames if f.has_wcs)
    calib = NightCalibration(
        night_id=manifest.get("night_id", night_dir.name),
        sensor=manifest.get("sensor"),
        site=manifest.get("site"),
        n_frames_total=n_total,
        n_frames_with_photometry=len(frames),
        n_frames_with_wcs=n_with_wcs,
        frames=frames,
    )
    calib.zp_per_filter = _summarize_zp(frames)
    calib.extinction_per_filter = _fit_extinction(frames)
    calib.limiting_mag_p50_per_filter = _summarize_limiting_mag(frames, "limiting_magnitude_50")
    calib.limiting_mag_p90_per_filter = _summarize_limiting_mag(frames, "limiting_magnitude_90")

    logger.info(
        "NightCalibration %s: %d/%d frames had photometry, %d had WCS; "
        "filters with ZP: %s; extinction fits: %s",
        calib.night_id, calib.n_frames_with_photometry, calib.n_frames_total,
        calib.n_frames_with_wcs,
        sorted(calib.zp_per_filter.keys()),
        sorted(calib.extinction_per_filter.keys()),
    )
    return calib


# --- persistence + plots ------------------------------------------------------


def save_calibration(calib: NightCalibration, output_dir: str | Path) -> Path:
    """Write the aggregated calibration JSON. Returns the file path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "night_calibration.json"
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)
    logger.info("Wrote %s", path)
    return path


def plot_calibration(
    calib: NightCalibration, output_dir: str | Path
) -> list[Path]:
    """Render the calibration plot set. Quietly skips plots that have no data.

    matplotlib is imported lazily so this module loads cheaply when only the
    aggregation is needed.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 1) Extinction curve: per-star ZP offset (m_cat − m_inst) vs airmass —
    #    gray cloud + airmass-binned medians + linear fit whose slope is −k.
    #    Sidereal frames only (rate-track photometry is unreliable for ZP).
    #    Runs predating stars_zp_offset retention have no per-star offsets;
    #    those fall back to one point per frame (the frame ZP) with the
    #    per-filter Bouguer fit overlaid.
    # Mirror the frame-ZP selection (zp_min_snr, default 20): below it,
    # aperture fluxes at faint catalog positions are contaminated upward and
    # drag the offsets high. Normalize by exposure so offsets match the frame
    # ZP convention (m + 2.5·log10(flux/texp)) — stars_zp_offset itself is
    # m_cat − m_inst with m_inst = −2.5·log10(flux), no texp term.
    ext_min_snr = 20.0
    star_ext: list[tuple[float, float]] = []  # (airmass, per-star ZP)
    for f in _zp_frames(calib.frames):
        if f.airmass is None or not f.stars_zp_offset or not f.exposure_time:
            continue
        texp_term = 2.5 * math.log10(f.exposure_time)
        star_ext.extend(
            (f.airmass, off - texp_term)
            for m, off, snr, iso in zip(f.stars_mag, f.stars_zp_offset,
                                        f.stars_snr, _isolated_flags(f))
            if off is not None and snr >= ext_min_snr and iso
            and _snr_consistent(snr, m, f)
        )
    fig, ax = None, None
    if len(star_ext) >= 10:
        fig, ax = plt.subplots(figsize=(10, 7))
        airmasses = np.array([p[0] for p in star_ext])
        offsets = np.array([p[1] for p in star_ext])
        # Clip gross outliers (mismatched/blended stars) before plotting/fit.
        keep = np.abs(offsets - offsets.mean()) <= 3 * offsets.std()
        airmasses, offsets = airmasses[keep], offsets[keep]
        ax.scatter(airmasses, offsets, alpha=0.3, s=10, color="lightgray",
                   label="Individual stars")
        bin_edges = np.arange(airmasses.min(), airmasses.max() + 0.2, 0.2)
        centers, medians, err_lo, err_hi = [], [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = offsets[(airmasses >= lo) & (airmasses < hi)]
            if len(in_bin) < 3:
                continue
            med = float(np.median(in_bin))
            centers.append((lo + hi) / 2)
            medians.append(med)
            err_lo.append(med - float(np.percentile(in_bin, 16)))
            err_hi.append(float(np.percentile(in_bin, 84)) - med)
        if centers:
            ax.errorbar(
                centers, medians, yerr=[err_lo, err_hi], fmt="o",
                color="black", markersize=7, capsize=4, capthick=1.5,
                elinewidth=1.5, alpha=0.85,
                label="Binned data (median ± 1σ percentiles)",
            )
        # offset = m0 − k·X  →  fitted slope = −k.
        slope, intercept = np.polyfit(airmasses, offsets, 1)
        line_x = np.linspace(airmasses.min(), airmasses.max(), 50)
        ax.plot(line_x, slope * line_x + intercept, "r-", linewidth=2,
                alpha=0.8, label=f"Extinction: k={-slope:.3f} mag/airmass")
        ax.set_ylabel(r"per-star ZP (m$_{cat}$ + 2.5·log$_{10}$(flux/t$_{exp}$)) [mag]")
        ax.set_title(f"{calib.night_id}: extinction curve ({len(airmasses)} "
                     f"isolated stars, sidereal, SNR≥{ext_min_snr:.0f})")
    else:
        # Frame-level fallback: ZP per frame, per-filter Bouguer fit overlay.
        zp_pts = [(f.airmass, f.zero_point, f.filter_name or "unknown")
                  for f in _zp_frames(calib.frames)
                  if f.airmass is not None and f.zero_point is not None]
        if zp_pts:
            fig, ax = plt.subplots(figsize=(10, 7))
            filters = sorted({p[2] for p in zp_pts})
            cmap = plt.cm.viridis(np.linspace(0, 0.85, max(len(filters), 1)))
            for color, filt in zip(cmap, filters):
                xs = [p[0] for p in zp_pts if p[2] == filt]
                ys = [p[1] for p in zp_pts if p[2] == filt]
                ax.scatter(xs, ys, label=f"{filt} (n={len(xs)})", s=12,
                           alpha=0.6, color=color)
                fit = calib.extinction_per_filter.get(filt)
                if fit:
                    line_x = np.linspace(min(xs), max(xs), 50)
                    ax.plot(
                        line_x, fit.m0 - fit.k * line_x, color=color,
                        linewidth=1.5,
                        label=f"  k={fit.k:.3f}±{fit.k_err:.3f}, "
                              f"m0={fit.m0:.3f}±{fit.m0_err:.3f}",
                    )
            ax.set_ylabel("zero point (instrumental → catalog mag)")
            ax.set_title(f"{calib.night_id}: Bouguer extinction (per-frame)")
    if fig is not None:
        ax.set_xlabel("Airmass")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        # Twin top axis: airmass → altitude (alt = arcsin(1/X)).
        ax2 = ax.twiny()
        ticks = np.array([t for t in ax.get_xticks() if t >= 1.0])
        if len(ticks):
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(
                [f"{math.degrees(math.asin(1.0 / t)):.0f}°" for t in ticks])
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel("Altitude")
        paths.append(_save(fig, output_dir / "extinction_curve.png"))

    # 2) Limiting magnitude (50%) distribution per filter (sidereal frames only —
    #    rate-track completeness is unreliable; see _summarize_limiting_mag)
    lim_data = [(f.limiting_magnitude_50, f.filter_name or "unknown")
                for f in _zp_frames(calib.frames)
                if f.limiting_magnitude_50 is not None]
    if lim_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        filters = sorted({d[1] for d in lim_data})
        for filt in filters:
            xs = [d[0] for d in lim_data if d[1] == filt]
            ax.hist(xs, bins=30, alpha=0.5, label=f"{filt} (n={len(xs)})")
        ax.set_xlabel("limiting magnitude (50% completeness)")
        ax.set_ylabel("number of frames")
        ax.set_title(f"{calib.night_id}: limiting magnitude distribution")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        paths.append(_save(fig, output_dir / "limiting_magnitude_hist.png"))

    # 3) Az/Alt coverage polar plot
    aa = [(f.azimuth_deg, f.altitude_deg, f.timestamp)
          for f in calib.frames
          if f.azimuth_deg is not None and f.altitude_deg is not None]
    if aa:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ts0 = min(t for _, _, t in aa if t is not None) if any(t for _, _, t in aa) else None
        if ts0 is not None:
            colors = [
                (t - ts0).total_seconds() if t is not None else 0
                for _, _, t in aa
            ]
        else:
            colors = [0] * len(aa)
        thetas = [math.radians(p[0]) for p in aa]
        rs = [90 - p[1] for p in aa]  # zenith distance — center = zenith
        sc = ax.scatter(thetas, rs, c=colors, s=12, cmap="plasma", alpha=0.7)
        ax.set_ylim(0, 90)
        ax.set_yticks([15, 30, 45, 60, 75])
        ax.set_yticklabels([f"{90-r}°" for r in [15, 30, 45, 60, 75]])
        ax.set_title(f"{calib.night_id}: Az/Alt coverage  (n={len(aa)})", pad=20)
        if ts0 is not None:
            cb = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
            cb.set_label("seconds since first frame")
        paths.append(_save(fig, output_dir / "alt_az_coverage.png"))

    # 4) ZP drift over the night (sidereal frames only)
    drift = [(f.timestamp, f.zero_point, f.filter_name or "unknown")
             for f in _zp_frames(calib.frames)
             if f.timestamp is not None and f.zero_point is not None]
    if drift:
        fig, ax = plt.subplots(figsize=(10, 5))
        filters = sorted({d[2] for d in drift})
        for filt in filters:
            xs = [d[0] for d in drift if d[2] == filt]
            ys = [d[1] for d in drift if d[2] == filt]
            ax.scatter(xs, ys, label=f"{filt} (n={len(xs)})", s=10, alpha=0.6)
        ax.set_xlabel("UTC time")
        ax.set_ylabel("zero point")
        ax.set_title(f"{calib.night_id}: zero point drift")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        paths.append(_save(fig, output_dir / "zp_drift.png"))

    # 5) Limiting magnitude vs exposure time (colored by airmass)
    #    Shows the depth/exposure trade-off and how much extra exposure buys
    #    at given airmass.
    depth_pts = [(f.exposure_time, f.limiting_magnitude_50, f.airmass)
                 for f in calib.frames
                 if f.exposure_time and f.limiting_magnitude_50]
    if depth_pts:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [p[0] for p in depth_pts]
        ys = [p[1] for p in depth_pts]
        cs = [p[2] if p[2] is not None else 0.0 for p in depth_pts]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=24, alpha=0.8,
                        edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("exposure time (s)")
        ax.set_ylabel("limiting magnitude (50% completeness)")
        ax.set_title(f"{calib.night_id}: depth vs exposure time")
        ax.grid(True, alpha=0.3)
        if any(c > 0 for c in cs):
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("airmass")
        paths.append(_save(fig, output_dir / "depth_vs_exposure.png"))

    # 6) Search rate vs magnitude, per STAR (sidereal frames only — rate-track
    #    aperture photometry is unreliable). For each measured star, scale its
    #    observed SNR to the exposure needed for SNR=6 (background-limited:
    #    SNR ∝ √t → t_req = t·(6/snr)²), clamp at a minimum exposure, add
    #    readout, and convert to sky area covered per hour at that cadence.
    #    Bright stars hit the exposure floor → the flat ceiling on the left;
    #    the roll-off to the right is the depth/coverage trade-off.
    target_snr = 6.0
    min_exposure_s = 0.1
    readout_s = 1.0
    # Stars below this measured SNR are forced-aperture noise at catalog
    # positions (the arrays retain every Gaia position with snr > 0, far past
    # the detection limit) — Eddington-biased, and the (6/snr)² extrapolation
    # is meaningless there.
    min_meas_snr = 5.0
    star_pts: list[tuple[float, float]] = []  # (catalog mag, deg²/hr at SNR=6)
    lim50s: list[float] = []
    for f in _zp_frames(calib.frames):
        if not f.exposure_time or not f.fov_sq_deg or not f.stars_mag:
            continue
        if f.limiting_magnitude_50 is not None:
            lim50s.append(f.limiting_magnitude_50)
        for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f)):
            if not iso:
                continue  # blended: the measurement belongs to the neighbor
            if s < min_meas_snr:
                # Non-detection on THIS frame: contributes zero search rate,
                # so faint bins honestly read "mostly undetectable" (deep
                # frames pull the median up) instead of showing only the
                # surviving measurements — high, and wildly variable.
                star_pts.append((m, 0.0))
            elif _snr_consistent(s, m, f):
                t_req = max(f.exposure_time * (target_snr / s) ** 2,
                            min_exposure_s)
                star_pts.append((m, f.fov_sq_deg / (t_req + readout_s) * 3600.0))
            # else: flux inconsistent with catalog mag (wings/variables/bad
            # cross-match) — not a measurement of this star; drop entirely.
    if star_pts:
        fig, ax = plt.subplots(figsize=(10, 7))
        mags = np.array([p[0] for p in star_pts])
        rates = np.array([p[1] for p in star_pts])
        ax.scatter(mags, rates, alpha=0.3, s=10, color="lightgray",
                   label="Individual stars")
        # Binned medians with asymmetric 16/84-percentile error bars.
        bin_width = 0.5
        bin_edges = np.arange(math.floor(mags.min() / bin_width) * bin_width,
                              mags.max() + bin_width, bin_width)
        centers, medians, err_lo, err_hi = [], [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = rates[(mags >= lo) & (mags < hi)]
            if len(in_bin) < 5:
                continue
            med = float(np.median(in_bin))
            centers.append((lo + hi) / 2)
            medians.append(med)
            err_lo.append(med - float(np.percentile(in_bin, 16)))
            err_hi.append(float(np.percentile(in_bin, 84)) - med)
        if centers:
            ax.errorbar(
                centers, medians, yerr=[err_lo, err_hi], fmt="o",
                color="black", markersize=7, capsize=4, capthick=1.5,
                elinewidth=1.5, alpha=0.85,
                label="Binned data (median ± 1σ percentiles)",
            )
        if lim50s:
            med_lim = float(np.median(lim50s))
            ax.axvline(med_lim, color="firebrick", linestyle="--", linewidth=1.5,
                       alpha=0.8, label=f"median lim. mag (50%) = {med_lim:.1f}")
        ax.set_xlabel("Apparent Magnitude (Catalog)")
        ax.set_ylabel(f"Search Rate (deg²/hour at SNR={target_snr:.0f})")
        ax.set_title(f"{calib.night_id}: search rate vs magnitude "
                     f"({len(star_pts)} isolated stars, sidereal, "
                     f"SNR≥{min_meas_snr:.0f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        paths.append(_save(fig, output_dir / "search_rate.png"))

    # 7) n_stars detected vs altitude (or airmass when alt unavailable)
    counts = [(f.altitude_deg, f.n_stars, f.exposure_time,
               f.filter_name or "unknown")
              for f in calib.frames
              if f.n_stars and f.altitude_deg is not None]
    if counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Normalize by exposure to compare across heterogeneous exposures.
        xs = [c[0] for c in counts]
        ys = [c[1] / c[2] if c[2] else c[1] for c in counts]
        filters = sorted({c[3] for c in counts})
        for filt in filters:
            mask = [c[3] == filt for c in counts]
            ax.scatter(
                [x for x, m in zip(xs, mask) if m],
                [y for y, m in zip(ys, mask) if m],
                label=f"{filt} (n={sum(mask)})", s=20, alpha=0.7,
            )
        ax.set_xlabel("altitude (deg)")
        ax.set_ylabel("stars detected per second")
        ax.set_yscale("log")
        ax.set_title(f"{calib.night_id}: detection rate vs altitude")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        paths.append(_save(fig, output_dir / "detection_rate_vs_altitude.png"))

    # 8) SNR vs exposure time, one errorbar series per 1-mag bin (sidereal
    #    frames only). For each (mag bin, standard exposure) cell with ≥3
    #    stars: median SNR ± 16/84 percentiles. Series connect across
    #    exposures so the √t scaling (or departure from it) is readable.
    snr_pts = []  # (exptime, mag, snr)
    for f in _zp_frames(calib.frames):
        if not f.exposure_time or not f.stars_mag:
            continue
        # Same forced-photometry guards as the search-rate plot: below
        # min_meas_snr the "measurement" is background noise at a catalog
        # position (median per bin = pure Eddington bias), and non-isolated
        # stars report their brighter neighbor's flux.
        snr_pts.extend(
            (f.exposure_time, m, s)
            for m, s, iso in zip(f.stars_mag, f.stars_snr, _isolated_flags(f))
            if s >= min_meas_snr and iso and _snr_consistent(s, m, f)
        )
    if snr_pts:
        fig, ax = plt.subplots(figsize=(10, 7))
        exps = np.array([p[0] for p in snr_pts])
        mags = np.array([p[1] for p in snr_pts])
        snrs = np.array([p[2] for p in snr_pts])
        # Standard exposure grid: whole seconds spanning the data, matched
        # with ±0.5 s tolerance so e.g. 1.2 s frames land on the 1 s point.
        std_exps = np.arange(max(1, math.floor(exps.min())),
                             math.ceil(exps.max()) + 1)
        mag_lo = math.floor(mags.min())
        mag_hi = math.ceil(mags.max())
        bins = list(range(mag_lo, mag_hi))
        colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(bins), 1)))
        for color, lo in zip(colors, bins):
            in_bin = (mags >= lo) & (mags < lo + 1)
            if in_bin.sum() < 5:
                continue
            xs, meds, e_lo, e_hi = [], [], [], []
            for t in std_exps:
                sel = snrs[in_bin & (np.abs(exps - t) <= 0.5)]
                if len(sel) < 3:
                    continue
                med = float(np.median(sel))
                xs.append(t)
                meds.append(med)
                e_lo.append(med - float(np.percentile(sel, 16)))
                e_hi.append(float(np.percentile(sel, 84)) - med)
            if len(xs) < 2:
                continue
            ax.errorbar(
                xs, meds, yerr=[e_lo, e_hi], fmt="o-", color=color,
                alpha=0.75, linewidth=1.5, markersize=6, capsize=3,
                label=f"{lo + 0.5:.1f}",
            )
        ax.set_yscale("log")
        ax.set_xticks(std_exps)
        ax.set_xlabel("Exposure Time [seconds]")
        ax.set_ylabel("Signal to Noise Ratio (SNR)")
        ax.set_title(f"{calib.night_id}: SNR vs exposure time (sidereal)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9, title=r"m$_G$")
        paths.append(_save(fig, output_dir / "snr_vs_exposure_by_magnitude.png"))

    return paths


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


# --- CLI hook -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI: ``python -m senpai.engine.observability.calibration <night_dir>``.

    The same code is invokable from ``senpai-burr calibrate`` (Phase 3 wiring)."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate per-batch SenpaiRun JSONs into a night calibration."
    )
    parser.add_argument(
        "night_dir",
        help="Processed-night dir (output of `senpai-burr night ...`); "
             "must contain manifest.json.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output dir for calibration JSON + plots (default: <night_dir>/calibration/).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot rendering (faster; matplotlib not required).",
    )
    args = parser.parse_args(argv)

    night_dir = Path(args.night_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (night_dir / "calibration")

    calib = analyze_night(night_dir)
    save_calibration(calib, out_dir)
    if not args.no_plots:
        plot_calibration(calib, out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
