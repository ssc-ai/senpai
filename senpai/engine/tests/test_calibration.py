"""Tests for senpai.engine.observability.calibration — per-frame extraction
plus the aggregation engine (ZP percentiles + Bouguer extinction fit).
"""

from __future__ import annotations

import math

import pytest

from datetime import datetime, timedelta, timezone

from types import SimpleNamespace

from senpai.engine.observability.calibration import (
    FramePhoto,
    FrameTiming,
    _airmass,
    _data_gain,
    _empirical_overhead,
    _extract_frame_photo,
    _extract_frame_timing,
    _fit_extinction,
    _fit_slew_model,
    _percentile,
    _summarize_limiting_mag,
    _summarize_zp,
)

# --- airmass ------------------------------------------------------------------


class TestAirmass:
    def test_zenith(self):
        assert _airmass(90.0) == pytest.approx(1.0)

    def test_45deg(self):
        assert _airmass(45.0) == pytest.approx(math.sqrt(2.0), rel=1e-6)

    def test_clipped_at_low_altitude(self):
        # Plane-parallel sec(z) diverges near the horizon; we explicitly
        # return None below 3° to keep plots/fits stable.
        assert _airmass(2.0) is None
        assert _airmass(None) is None


# --- per-frame extract --------------------------------------------------------


class TestExtractFramePhoto:
    def _frame(self, **overrides):
        base = {
            "index": 0,
            "timestamp": "2026-05-27T07:00:00+00:00",
            "frame_metadata": {
                "observation_time": "2026-05-27T07:00:00+00:00",
                "exposure_time_seconds": 5.0,
                "observation_filter": "V",
                "track_mode": "sidereal",
            },
            "starfield": {"wcs_metadata": {
                "RA_center_deg": 180.0, "Dec_center_deg": 30.0,
            }},
            "photometry_summary": {
                "n_stars": 100, "n_quality": 80,
                "median_snr": 12.0, "median_background": 50.0,
                "limiting_magnitude": 19.0,
                "limiting_magnitude_50": 19.0, "limiting_magnitude_90": 18.0,
                "zero_point": 24.5, "zero_point_err": 0.03,
            },
        }
        base.update(overrides)
        return base

    def test_happy_path_fills_all_fields(self):
        site = {"latitude": 20.0, "longitude": -156.0, "altitude_km": 0.1}
        fp = _extract_frame_photo(self._frame(), "batch1", site, "sidereal")
        assert fp is not None
        assert fp.batch_id == "batch1"
        assert fp.zero_point == 24.5
        assert fp.filter_name == "V"
        # alt/az should populate via astropy when site is present.
        assert fp.altitude_deg is not None
        assert fp.azimuth_deg is not None
        # airmass is derived from altitude (when > 3°)
        if fp.altitude_deg > 3.0:
            assert fp.airmass is not None

    def test_returns_none_without_photometry(self):
        frame = self._frame()
        frame["photometry_summary"] = None
        assert _extract_frame_photo(frame, "b", None, "sidereal") is None

    def test_handles_missing_wcs(self):
        frame = self._frame()
        frame["starfield"] = None
        fp = _extract_frame_photo(frame, "b", None, "sidereal")
        assert fp is not None
        assert fp.zero_point == 24.5
        assert not fp.has_wcs
        assert fp.altitude_deg is None
        assert fp.airmass is None

    def test_handles_missing_site_gracefully(self):
        frame = self._frame()
        fp = _extract_frame_photo(frame, "b", site=None, track_mode_default="sidereal")
        assert fp is not None
        assert fp.has_wcs  # RA/Dec present even though we can't compute alt/az
        assert fp.altitude_deg is None
        assert fp.azimuth_deg is None

    def test_multiband_zps_extracted(self):
        frame = self._frame()
        frame["photometry_summary"]["multiband_calibration"] = {
            "bands": {
                "g": {"zero_point": 24.6},
                "r": {"zero_point": 24.3},
            },
        }
        fp = _extract_frame_photo(frame, "b", None, "sidereal")
        assert fp.multiband_zps == {"g": 24.6, "r": 24.3}

    def test_aperture_geometry_lifted_from_summary(self):
        frame = self._frame()
        frame["photometry_summary"]["aperture_geometry"] = {
            "shape": "circle", "fwhm_px": 3.5,
            "aperture_radius_px": 7.0, "bg_inner_px": 10.5, "bg_outer_px": 17.5,
        }
        fp = _extract_frame_photo(frame, "b", None, "sidereal")
        assert fp.aperture_geometry["shape"] == "circle"
        assert fp.aperture_geometry["aperture_radius_px"] == 7.0

    def test_aperture_geometry_none_on_legacy_summary(self):
        # A summary predating aperture_geometry retention leaves the field None.
        fp = _extract_frame_photo(self._frame(), "b", None, "sidereal")
        assert fp.aperture_geometry is None


# --- percentile + ZP summary --------------------------------------------------


class TestPercentile:
    def test_endpoints(self):
        assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0
        assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0

    def test_interpolation(self):
        # idx = 0.5 * 2 = 1.0 → exact index 1 → value 2.0
        assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0
        # idx = 0.25 * 2 = 0.5 → halfway between 1.0 and 2.0 → 1.5
        assert _percentile([1.0, 2.0, 3.0], 0.25) == 1.5

    def test_empty_returns_nan(self):
        import math
        assert math.isnan(_percentile([], 0.5))


def _fp(filter_name: str | None, zp: float | None, **kw) -> FramePhoto:
    # ZP/extinction aggregation is sidereal-only, so fixtures default to
    # sidereal; pass track_mode="rate" to exercise the exclusion.
    return FramePhoto(
        batch_id="b", frame_index=0, timestamp=None,
        track_mode=kw.get("track_mode", "sidereal"),
        filter_name=filter_name, exposure_time=None,
        zero_point=zp, zero_point_err=kw.get("zp_err"),
        limiting_magnitude_50=kw.get("lim50"),
        limiting_magnitude_90=kw.get("lim90"),
        median_snr=None, median_background=None,
        n_stars=None, n_quality=None,
        ra_center_deg=None, dec_center_deg=None,
        altitude_deg=kw.get("alt"),
        azimuth_deg=None,
        airmass=kw.get("airmass"),
    )


class TestSummarizeZP:
    def test_groups_by_filter(self):
        frames = [
            _fp("V", 24.1), _fp("V", 24.5), _fp("V", 24.3),
            _fp("B", 23.0), _fp("B", 23.4),
            _fp("V", None),  # ignored
        ]
        out = _summarize_zp(frames)
        assert out["V"].n == 3 and out["B"].n == 2
        assert out["V"].median == pytest.approx(24.3)

    def test_unknown_filter_bucket(self):
        frames = [_fp(None, 24.0), _fp(None, 24.4)]
        out = _summarize_zp(frames)
        assert "unknown" in out and out["unknown"].n == 2

    def test_excludes_rate_frames(self):
        # Rate-track photometry is unreliable for ZP and must be excluded.
        frames = [
            _fp("V", 24.3),
            _fp("V", 24.5),
            _fp("V", 19.0, track_mode="rate"),  # streaked → must not count
        ]
        out = _summarize_zp(frames)
        assert out["V"].n == 2
        assert out["V"].median == pytest.approx(24.4)

    def test_skips_when_no_zp(self):
        frames = [_fp("V", None), _fp("V", None)]
        out = _summarize_zp(frames)
        assert "V" not in out


# --- Bouguer extinction fit ---------------------------------------------------


class TestFitExtinction:
    def test_recovers_known_line(self):
        # zp = 24.5 - 0.20 * airmass  (no noise)
        airmasses = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
        zps = [24.5 - 0.20 * a for a in airmasses]
        frames = [_fp("V", z, airmass=a) for z, a in zip(zps, airmasses, strict=True)]
        out = _fit_extinction(frames)
        assert "V" in out
        fit = out["V"]
        assert fit.k == pytest.approx(0.20, abs=1e-9)
        assert fit.m0 == pytest.approx(24.5, abs=1e-9)
        assert fit.n == 6
        assert fit.airmass_range == (1.0, 3.0)

    def test_skips_filters_with_too_few_points(self):
        frames = [
            _fp("V", 24.0, airmass=1.0),
            _fp("V", 23.8, airmass=1.5),  # only 2 points → skip
        ]
        assert _fit_extinction(frames) == {}

    def test_ignores_frames_without_airmass(self):
        # 2 with airmass + 1 without → still only 2 valid → skipped
        frames = [
            _fp("V", 24.5, airmass=1.0),
            _fp("V", 24.3, airmass=2.0),
            _fp("V", 24.0, airmass=None),
        ]
        assert _fit_extinction(frames) == {}

    def test_skips_degenerate_airmass_range(self):
        # All frames at ~one pointing → airmass span below the guard → no fit.
        frames = [
            _fp("V", 24.50, airmass=1.330),
            _fp("V", 24.49, airmass=1.331),
            _fp("V", 24.51, airmass=1.332),
        ]
        assert _fit_extinction(frames) == {}

    def test_excludes_rate_frames(self):
        # Even with a good airmass spread, rate frames must not drive extinction.
        frames = [
            _fp("V", 24.5, airmass=1.0, track_mode="rate"),
            _fp("V", 24.3, airmass=2.0, track_mode="rate"),
            _fp("V", 24.1, airmass=3.0, track_mode="rate"),
        ]
        assert _fit_extinction(frames) == {}


# --- limiting mag summary -----------------------------------------------------


def test_summarize_limiting_mag_per_filter():
    frames = [
        _fp("V", 24.0, lim50=19.0),
        _fp("V", 24.0, lim50=19.4),
        _fp("V", 24.0, lim50=19.2),
        _fp("B", 23.0, lim50=18.8),
    ]
    out = _summarize_limiting_mag(frames, "limiting_magnitude_50")
    assert out["V"] == pytest.approx(19.2)  # median
    assert out["B"] == pytest.approx(18.8)


def test_summarize_limiting_mag_excludes_rate_frames():
    """Rate-track completeness is unreliable, so the night's authoritative
    limiting mag must ignore rate frames (a deep, spurious rate value must not
    pull the median)."""
    frames = [
        _fp("V", 24.0, lim50=19.0),
        _fp("V", 24.0, lim50=19.2),
        _fp("V", 24.0, lim50=23.5, track_mode="rate"),  # spurious deep rate value
    ]
    out = _summarize_limiting_mag(frames, "limiting_magnitude_50")
    assert out["V"] == pytest.approx(19.1)  # median of the two sidereal only


# --- inter-frame overhead (slew/settle) model ---------------------------------


class TestExtractFrameTiming:
    """Timing+pointing is kept for EVERY frame (incl. non-photometric ones) via
    the commanded boresight, since slew/settle is mount mechanics independent of
    photometry — see _extract_frame_timing."""

    def _frame(self, **overrides):
        base = {
            "timestamp": "2026-05-27T07:00:00+00:00",
            "frame_metadata": {
                "exposure_time_seconds": 5.0,
                "track_mode": "rate",
                "boresight_ra_degrees": 180.0,
                "boresight_dec_degrees": 30.0,
            },
            "photometry_summary": None,  # non-fit frame: no photometry at all
        }
        base.update(overrides)
        return base

    def test_keeps_frame_without_photometry(self):
        # _extract_frame_photo would drop this frame; timing must still keep it.
        assert _extract_frame_photo(self._frame(), "b", None, "rate") is None
        ret = _extract_frame_timing(self._frame(), "rate")
        assert ret is not None
        timing, ra, dec = ret
        assert (ra, dec) == (180.0, 30.0)
        assert timing.exposure_time == 5.0
        assert timing.track_mode == "rate"
        assert timing.altitude_deg is None  # filled later, vectorized
        assert timing.fov_sq_deg is None    # no WCS solve → no measured FoV

    def test_none_without_boresight(self):
        frame = self._frame()
        frame["frame_metadata"] = {"exposure_time_seconds": 5.0}
        assert _extract_frame_timing(frame, "rate") is None

    def test_none_without_exposure(self):
        frame = self._frame()
        frame["frame_metadata"]["exposure_time_seconds"] = None
        assert _extract_frame_timing(frame, "rate") is None

    def test_fov_only_from_solved_frames(self):
        # Only when a WCS solved does the record carry a (measured) FoV — used
        # for the contiguous-grid step, which must come from good frames only.
        frame = self._frame()
        frame["starfield"] = {"wcs_metadata": {
            "x_fov_degrees": 2.0, "y_fov_degrees": 1.5}}
        timing, _ra, _dec = _extract_frame_timing(frame, "sidereal")
        assert timing.fov_sq_deg == pytest.approx(3.0)


def _timing(t0, secs, exposure, alt, az, fov=None):
    return FrameTiming(
        timestamp=t0 + timedelta(seconds=secs),
        exposure_time=exposure, track_mode="sidereal",
        altitude_deg=alt, azimuth_deg=az, fov_sq_deg=fov,
    )


class TestEmpiricalOverhead:
    """Fallback overhead when the full two-regime fit can't be constrained:
    the night's observed cadence, not a flat guess."""

    def test_no_pairs_returns_none(self):
        # Fewer than two placeable frames → no pairs → caller uses its default.
        assert _empirical_overhead([]) == (None, "")

    def test_prefers_observed_slew_cadence(self):
        # A run that parks on two fields, alternating with a big (~90°) slew at a
        # steady ~9 s overhead. Too few distinct distances for the line fit, but
        # the median slewed-pair overhead is exactly what we want as a fallback.
        t0 = datetime(2026, 5, 27, 7, tzinfo=timezone.utc)
        timings, t = [], 0.0
        for i in range(20):
            alt, az = (60.0, 10.0) if i % 2 == 0 else (60.0, 100.0)
            timings.append(_timing(t0, t, 5.0, alt, az))
            t += 14.0  # 5 s exposure + ~9 s overhead
        overhead, label = _empirical_overhead(timings)
        assert overhead == pytest.approx(9.0, abs=0.5)
        assert "slew" in label

    def test_full_fit_unusable_falls_back_not_crashes(self):
        # Same single-distance data: the strict fit can't constrain the slope.
        t0 = datetime(2026, 5, 27, 7, tzinfo=timezone.utc)
        timings, t = [], 0.0
        for i in range(20):
            alt, az = (60.0, 10.0) if i % 2 == 0 else (60.0, 100.0)
            timings.append(_timing(t0, t, 5.0, alt, az))
            t += 14.0
        assert _fit_slew_model(timings) is None
        overhead, _label = _empirical_overhead(timings)
        assert overhead and overhead > 1.0


class TestFitSlewModelOnTimings:
    def test_fits_two_regime_model_from_timings(self):
        # Synthetic night: repeat exposures (readout-only) interleaved with slews
        # of varied separation at a known rate, so the readout floor and a rising
        # slew line are both well sampled across several distance bins. The slew
        # is in ALTITUDE at fixed azimuth, where Δalt equals the on-sky
        # separation exactly (no cos(alt) compression), and the slew time goes in
        # the gap BEFORE the slewed frame — Δt(i→i+1)=exposure+readout+slew.
        t0 = datetime(2026, 5, 27, 7, tzinfo=timezone.utc)
        rate, readout, exposure, az = 10.0, 2.0, 5.0, 10.0  # deg/s, s, s, deg
        base, seps = 45.0, [0.5, 1.0, 3.0, 6.0, 12.0, 30.0]
        # alt sequence: two repeats at base (a readout-only pair to anchor the
        # floor) then a slew of `sep` to base+sep.
        alts = [base]
        for _block in range(8):
            for sep in seps:
                alts += [base, base, base + sep]
        timings, t = [_timing(t0, 0.0, exposure, alts[0], az, fov=4.0)], 0.0
        for prev, alt in zip(alts, alts[1:]):
            t += exposure + readout + abs(alt - prev) / rate
            timings.append(_timing(t0, t, exposure, alt, az, fov=4.0))
        d = _fit_slew_model(timings)
        assert d is not None
        assert d["readout_s"] == pytest.approx(readout, abs=0.5)
        assert d["slew_rate_deg_s"] == pytest.approx(rate, rel=0.2)
        # grid step = sqrt(median fov) = 2°, so cadence overhead ≈ bias + 2/rate.
        assert d["fov_width_deg"] == pytest.approx(2.0, abs=0.01)
        assert d["grid_overhead_s"] > d["readout_s"]


# --- detector gain plot data (photon transfer from sky) ----------------------


class TestGainPlotData:
    def _frame(self, sky, gain):
        f = _fp("V", 24.0)
        f.sky_adu = sky
        f.gain_e_per_adu = gain
        return f

    def _calib(self, frames):
        return SimpleNamespace(frames=frames)

    def test_collects_pairs_and_summarizes(self):
        frames = [self._frame(s, g) for s, g in
                  [(400.0, 2.0), (800.0, 2.1), (1200.0, 1.9), (1600.0, 2.0)]]
        d = _data_gain(self._calib(frames))
        assert d["n"] == 4
        assert len(d["sky_adu"]) == 4 and len(d["gain"]) == 4
        assert d["median"] == pytest.approx(2.0, abs=0.05)
        assert d["std"] >= 0.0

    def test_skips_frames_missing_gain_or_sky(self):
        frames = [
            self._frame(500.0, 2.0),
            self._frame(600.0, None),   # no gain -> skipped
            self._frame(None, 2.1),     # no sky  -> skipped
            self._frame(700.0, 1.9),
            self._frame(800.0, 2.2),
        ]
        d = _data_gain(self._calib(frames))
        assert d["n"] == 3

    def test_returns_none_with_too_few_points(self):
        frames = [self._frame(500.0, 2.0), self._frame(600.0, 2.1)]
        assert _data_gain(self._calib(frames)) is None
