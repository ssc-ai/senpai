"""Application configuration: pydantic-settings models and the process-global singleton.

The configuration is a pydantic-settings tree rooted at :class:`AppConfig`, loaded from a
YAML file (either flat sections at the top level or nested under a single ``app:`` key) with
every field overridable through environment variables using ``__`` as the nesting delimiter
(e.g. ``ASTROMETRY__CPULIMIT_SECONDS=60``). Environment variables take precedence over the
YAML file.

``initialize_config`` populates the module-global singleton returned by ``get_config``;
:mod:`senpai.settings` exposes the same instance through a lazy ``settings`` proxy.
"""

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from senpai.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Global singleton instance
_config_instance: Optional["AppConfig"] = None

# YAML file consumed by the next AppConfig construction; set (and reset) by
# initialize_config so direct AppConfig(**kwargs) constructions read no file.
_yaml_file_for_next_init: Path | None = None


def load_yaml(path: Path) -> dict:
    """Load a YAML config file and return its ``app`` section.

    Retained for backwards compatibility with configs nested under an ``app:`` key;
    ``initialize_config`` accepts both nested and flat layouts.

    Args:
        path (Path): path to the YAML file.

    Returns:
        dict: the ``app`` section contents, or ``{}`` when the file is missing,
            malformed, or has no ``app`` key.
    """
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
            return data.get("app", {})
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}


def _load_config_mapping(path: Path) -> dict:
    """Load a YAML config file accepting both flat and ``app:``-nested layouts.

    Args:
        path (Path): path to the YAML file.

    Returns:
        dict: the config mapping (the ``app`` section when present, otherwise the whole
            document), or ``{}`` when the file is missing or malformed.
    """
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    app_section = data.get("app")
    if isinstance(app_section, dict):
        return app_section
    return data


class _YamlConfigSource(PydanticBaseSettingsSource):
    """Settings source reading the YAML file set by ``initialize_config``.

    Loads the whole mapping in one shot (flat or ``app:``-nested) rather than
    per-field, so ``__call__`` is overridden and ``get_field_value`` is unused.
    """

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path | None) -> None:
        """Store the target YAML path.

        Args:
            settings_cls (type[BaseSettings]): the settings class being configured.
            yaml_file (Path | None): YAML file to read, or None for no file source.
        """
        super().__init__(settings_cls)
        self._yaml_file = yaml_file

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ANN401
        """Unused per-field hook (the whole mapping is returned by ``__call__``).

        Args:
            field (Any): field being resolved (unused).
            field_name (str): name of the field (unused).

        Returns:
            tuple[Any, str, bool]: a "no value" sentinel triple.
        """
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the YAML mapping for this settings construction.

        Returns:
            dict[str, Any]: parsed config mapping, or ``{}`` when no file is set.
        """
        if self._yaml_file is None:
            return {}
        return _load_config_mapping(Path(self._yaml_file))


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"

    model_config = ConfigDict(frozen=True)


class PlottingConfig(BaseModel):
    """Plotting configuration."""

    debug: bool = Field(description="Debug Plots")
    review: bool = Field(description="Review Plots")
    photometry: bool = Field(default=False, description="Photometry Plots")
    psfs: bool = Field(
        default=False,
        description="Per-frame empirical PSF plots: a stacked-star PSF panel for "
        "sidereal frames and a stacked-streak panel for rate frames (small, "
        "~<1MB each; separate from the heavy `debug` kernel/CC plots). A little "
        ".npy stamp is saved alongside so the panels regenerate after the fact.",
        validation_alias=AliasChoices("psfs", "streak"),
    )
    paper_ready: bool = Field(
        default=False,
        description="Also emit a title-less '<name>_clean.<ext>' copy of each saved "
        "figure (PSF panels + night calibration/observability plots), for dropping "
        "into a paper where the caption replaces the on-figure title. The normal "
        "titled figure is still written.",
    )
    output_dir: str = Field(default=".", description="Directory debug plots are written to.")


class FastSolveConfig(BaseModel):
    """Native fast-solve tier settings (solver_mode 'tetra3'/'chain'); never read by 'dotnet'."""

    mirror_dir: str | None = Field(
        default=None,
        description="Local Gaia mirror directory (the T0 refine catalog; see astroeasy.catalog.mirror)",
    )
    tetra3_db_path: str | None = Field(
        default=None,
        description="tetra3 pattern database (.npz) for the T1 lost-in-space matcher",
    )
    sensor_profile: str | None = Field(
        default=None,
        description="Sensor profile YAML (measured geometry + acceptance-gate thresholds)",
    )


class AstrometryConfig(BaseModel):
    """Astrometry(.net) configuration."""

    solver_mode: Literal["dotnet", "tetra3", "chain", "senpai"] = Field(
        default="dotnet",
        description="Plate-solve engine: 'dotnet' = astrometry.net via astroeasy (the original "
        "path), 'tetra3' = native catalog tiers only (T0 refine + T1 pattern match, no "
        "astrometry.net required), 'chain' = full escalation cascade (native tiers first, "
        "astrometry.net backstop), 'senpai' = in-process solver (astrometry PyPI package; "
        "blind solve + same-cell verify + SIP-1 refine)",
    )
    fast_solve: FastSolveConfig = Field(
        default_factory=FastSolveConfig,
        description="Settings for the native fast-solve tiers (only read when solver_mode != 'dotnet')",
    )
    indices_series: str = Field(
        description="Indices series (5200/5200_LITE/5200_SENPAI/4100/5200_LITE_4100/4200/CUSTOM)"
    )
    indices_path: str = Field(description="Local indices path")
    max_sources: int = Field(description="Maximum number of sources to solve for")
    min_sources_for_attempt: int = Field(description="Minimum number of sources to attempt astrometry")
    min_width_degrees: float = Field(description="Minimum width in degrees")
    max_width_degrees: float = Field(description="Maximum width in degrees")
    cpulimit_seconds: int = Field(description="CPU limit in seconds")
    docker_image: str | None = Field(description="Docker image name")
    reduce_field_by_radius: float | None = Field(
        default=None,
        description="Reduce field to sources within this radius as % of image circle (null=full field, 1.0=circle contained by width/height)",
    )
    tweak_order: int = Field(
        default=3,
        description="SIP polynomial order for astrometry.net solve-field (2-5, higher for extreme pincushion distortion)",
    )
    sip_refit_order: int = Field(
        default=7,
        description="SIP order for post-solve refit using catalog stars (3-9, higher for extreme/complex distortion patterns)",
    )
    sip_refit_enabled: bool = Field(
        default=True,
        description="Enable SIP refit after initial solve using catalog stars for better edge distortion fitting",
    )
    search_radius_degrees: float = Field(
        default=5.0,
        description="Radius around boresight (if provided) for astrometry to search for a fit.",
    )
    source_extractor: str = Field(
        default="sextractor",
        description=(
            "Source extractor for plate solving. "
            "Options: 'sextractor' (SEP/SExtractor, recommended), "
            "'daofind' (photutils DAOFind), "
            "'image2xy' (astrometry.net bundled binary)."
        ),
    )
    sip_order: int = Field(
        default=3,
        description="SIP distortion polynomial order used by WCS refinement (0 disables SIP)",
    )
    min_logodds_threshold: float = Field(
        default=21.0,
        description="Minimum log-odds confidence for accepting a plate solution",
    )
    error_on_plate_solve_failure: bool = Field(
        default=False,
        description="If True, raise an exception if the image set has no WCS solution. "
        "If False, record the failure on the run and return blank output.",
    )
    require_complete_indices: bool = Field(
        default=False,
        description="If True, fail at startup if the star index is not present and complete "
        "(solver_mode 'senpai' only).",
    )
    release_index_cache_after_solve: bool = Field(
        default=True,
        description=(
            "After each in-process plate solve (solver_mode 'senpai'), drop the astrometry "
            "index files' page cache via posix_fadvise(POSIX_FADV_DONTNEED). The solver mmaps "
            "index tiles whose pages otherwise linger as file-backed page cache and accumulate "
            "toward the full index size across a run, so the process working set climbs like a "
            "leak even though the memory is reclaimable. Releasing after each solve keeps it "
            "flat; the cost is re-reading needed tiles from disk on the next solve."
        ),
    )


class StarCatalogConfig(BaseModel):
    """Star catalog configuration."""

    type: str = Field(description="Star catalog type")
    path: str | None = Field(
        default=None,
        description="Star catalog path (required for local catalogs like SSTRC7, not needed for online catalogs like SDSS)",
    )
    faint_limit: float | None = Field(
        default=18.0,
        description="Default faint magnitude limit for online catalogs (e.g., Gaia G); "
        "set to None to use the service default.",
    )
    max_stars_per_frame: int | None = Field(
        default=None,
        description="Cap on catalog stars returned per frame for callers that "
        "request the full catalog (max_stars=None). Applied as a magnitude-"
        "stratified subsample so completeness statistics survive; bounds the "
        "per-frame memory/CPU on dense galactic-plane fields (a 74k-star field "
        "needed ~30 GB/worker uncapped). None = unbounded.",
    )

    @model_validator(mode="after")
    def validate_catalog_config(self) -> "StarCatalogConfig":
        """Validate that path is provided for local catalogs but not required for online catalogs.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If a local catalog type is selected without a path.
        """
        # Online catalogs don't need a path
        if self.type in ["sdss", "gaia"]:
            return self  # Path can be None for online catalogs

        # Local catalogs require a path (gaia_local = trimmed Gaia mirror dir)
        if self.type in ["sstrc7", "gaia_local"] and self.path is None:
            raise ValueError(f"path is required for catalog type '{self.type}'")

        return self


class RuntimeConfig(BaseModel):
    """CLI runtime configuration."""

    run_id: str = Field(default="senpai", description="Run identifier")
    output_dir: str = Field(default=".", description="Output directory")
    save_processed_fits: bool = Field(
        default=True,
        description="Write per-frame *_processed.fits next to the results. "
        "Needed for decoupled replotting, but ~260 MB/frame on 8k sensors "
        "(~94% of a night's output) — full-night runs disable it via "
        "`senpai-burr night --no-processed-fits`.",
    )

    model_config = ConfigDict(frozen=False)  # Allow updates to Runtime config


class DetectionConfig(BaseModel):
    """Detection configuration."""

    detect: bool = Field(default=False, description="Detect point sources")
    detect_streaks: bool = Field(default=True, description="Run streak detection when detect=True")
    snr_threshold: float = Field(default=3.0, description="SNR threshold")
    verbose: bool = Field(default=False, description="Verbose mode")
    streak_correlation_radius_fwhm: float = Field(
        default=5.0, description="Match radius for cross-frame streak correlation, in FWHM units"
    )
    streak_angle_tolerance_deg: float = Field(
        default=15.0, description="Angle tolerance for cross-frame streak matching"
    )
    require_wcs_refinement: bool = Field(
        default=True,
        description="Only produce detections on frames where WCS refinement succeeded",
    )
    centroid_guard_mode: Literal["fwhm", "fixed", "none"] = Field(
        default="fwhm",
        description=(
            "Guard for the reported point-source position. The SEP sub-pixel centroid is "
            "reported unless it disagrees with the brightest pixel by more than a threshold, "
            "in which case the brightest pixel is reported instead (saturation/trail/blend "
            "protection). 'fwhm': threshold = centroid_guard_value * FWHM (PSF-relative); "
            "'fixed': threshold = centroid_guard_value pixels; 'none': always report the "
            "sub-pixel centroid."
        ),
    )
    centroid_guard_value: float = Field(
        default=0.4,
        description=(
            "Threshold for centroid_guard_mode: a multiple of the PSF FWHM ('fwhm') or an "
            "absolute pixel distance ('fixed'). Ignored when mode is 'none'."
        ),
    )
    sidereal_point_detections: bool = Field(
        default=True,
        description=(
            "Flag bright non-catalog point sources on solved sidereal frames as candidate "
            "detections. Disable for pipelines that report detections only on rate-track "
            "frames."
        ),
    )


class VariableKernelConfig(BaseModel):
    """Configuration for variable streak kernels driven by WCS distortion."""

    enable: bool = Field(
        default=False,
        description="Enable variable streak kernels for rate-track refinement when distortion is high",
    )
    angle_thresh_deg: float = Field(
        default=1.0,
        description="Minimum max_angle_variation_deg required to enable variable kernels",
    )
    length_thresh_fraction: float = Field(
        default=0.05,
        description="Minimum max_length_variation_fraction required to enable variable kernels",
    )
    diagnostics_max_stars: int = Field(
        default=16,
        description="Maximum number of stars to use for variable-kernel diagnostics plots",
    )
    diagnostics_grid_nx: int = Field(
        default=4,
        description="Number of grid points in x for kernel diagnostic mosaics",
    )
    diagnostics_grid_ny: int = Field(
        default=4,
        description="Number of grid points in y for kernel diagnostic mosaics",
    )


class StreakDetectionConfig(BaseModel):
    """Configuration for streak-specific detection options."""

    variable_kernel: VariableKernelConfig = Field(
        default_factory=VariableKernelConfig,
        description="Variable-kernel configuration for streak WCS refinement",
    )
    symmetric_border_removal: bool = Field(
        default=True,
        description="When removing border-crossing streaks before the rate-rate "
        "cross-correlation, also fill the counterpart region (blob translated by "
        "±expected drift) in the OTHER frame. Deleting a streak from only one "
        "frame breaks its correlation pair and lets a mis-pair of two different "
        "stars win the CC peak — the proven cause of reversed/aliased shifts. "
        "Falls back to per-frame removal when no drift estimate exists.",
    )
    reconcile_with_chain: bool = Field(
        default=True,
        description="Overrule a degenerate or deviant per-frame streak model "
        "(length==fwhm blob fits, lengths off by >reconcile_length_tolerance, "
        "axes misaligned with the drift) with chain-derived geometry "
        "(drift rate x exposure along the drift axis) once hops are solved",
    )
    reconcile_length_tolerance: float = Field(
        default=0.5,
        description="Fractional disagreement with rate x exposure beyond which "
        "the extracted streak length is replaced",
    )
    reconcile_angle_tolerance_deg: float = Field(
        default=25.0,
        description="Misalignment between streak axis and drift axis beyond "
        "which the extracted angle is replaced",
    )
    max_fwhm_for_streak_extraction: float = Field(
        default=10.0, description="Maximum FWHM allowed by extract_streak_dims_robust()."
    )


class ValidationConfig(BaseModel):
    """Configuration for box-based shift validation in rate tracking."""

    box_size: int = Field(
        default=11,
        description="Box size (pixels) around each star for lightweight validation",
    )
    n_random_trials: int = Field(default=8, description="Number of random shifts to test against proposed shift")
    random_radius_pixels: int = Field(default=40, description="Radius (pixels) for random shift generation")

    # Validation thresholds
    min_correlation_ratio: float = Field(
        default=0.98,
        description="Proposed shift must be within this ratio of best correlation (0.98 = within 2%)",
    )
    min_absolute_correlation: float = Field(
        default=0.6, description="Minimum absolute correlation required for validation"
    )
    lenient_absolute_correlation: float = Field(
        default=0.55,
        description="Lenient absolute correlation threshold when correlation ratio >= 0.93 (for cases with few stars)",
    )
    fewer_stars_correlation_ratio: float = Field(
        default=0.985,
        description="Stricter correlation ratio required when the proposed shift has "
        "fewer matched stars than the best trial. Was a hardcoded 0.99, which "
        "razor-thin-rejected correct shifts (ratio ~0.987) and fell through to a "
        "flipped shift; 0.985 keeps mild extra strictness over the base ratio.",
    )
    noise_correlation_ratio: float = Field(
        default=0.99,
        description="Strict correlation ratio required when >=3 random trials beat "
        "the proposed shift's star count (strong noise-correlation signal).",
    )
    noise_min_absolute_correlation: float = Field(
        default=0.70,
        description="Strict absolute-correlation floor for the same noise-signal case.",
    )
    max_validation_stars: int = Field(default=50, description="Maximum number of stars to use for validation")
    test_negated_shift: bool = Field(
        default=True,
        description="Also correlate the negated shift; reject the proposed shift if its "
        "negation correlates better (guards against the rate-rate direction ambiguity)",
    )
    negated_rejection_ratio: float = Field(
        default=1.05,
        description="Reject the proposed shift when corr(-shift) > ratio * corr(shift) "
        "and the shift is larger than a few pixels",
    )


class WCSValidationConfig(BaseModel):
    """Absolute post-refinement WCS validation.

    Tests for real star flux at the pixel positions the refined WCS predicts
    for the brightest catalog stars, against a random-position null and an
    offset control grid. Catches WCS solutions that are internally consistent
    but globally wrong (e.g. poisoned by a bad frame-to-frame shift), which
    the relative fallback-based checks cannot see.
    """

    enable: bool = Field(default=True, description="Run absolute WCS validation after refinement")
    n_stars: int = Field(default=30, description="Brightest in-bounds catalog stars to test")
    min_stars: int = Field(
        default=8,
        description="Minimum testable stars for a verdict; below this the result is indeterminate",
    )
    n_random: int = Field(default=500, description="Random positions for the null distribution")
    significance_percentile: float = Field(
        default=99.0, description="Null percentile a star's flux must exceed to count as significant"
    )
    min_frac_significant: float = Field(
        default=0.12,
        description="Minimum fraction of significant stars for the WCS to pass. "
        "Calibrated on the 2024 tako calsat set: genuinely poisoned WCS frames "
        "score <=0.04 and correct ones >=0.16 even under heavy moonlight.",
    )
    control_margin: float = Field(
        default=0.10,
        description="Pass additionally requires frac_significant >= control fraction + this margin",
    )
    control_offset_px: int = Field(
        default=300, description="Pixel offset applied to predictions for the control grid"
    )
    background_block_px: int = Field(
        default=32, description="Block size for the coarse median background model"
    )


class ChainGateConfig(BaseModel):
    """Consistency gate for the frame-to-frame shift chain.

    Under rate tracking the star drift rate (shift / time gap) is nearly
    constant across an observation, so a solved hop whose rate reverses
    direction or deviates grossly from the accepted-chain median is almost
    certainly a mis-solve. One such hop silently poisons the WCS of every
    frame chained beyond it.
    """

    enable: bool = Field(default=True, description="Enable the shift-chain consistency gate")
    min_history_hops: int = Field(
        default=2, description="Accepted rate-rate hops required before the gate activates"
    )
    max_rate_deviation_fraction: float = Field(
        default=0.5,
        description="Reject a hop whose drift rate deviates from the chain median by more than "
        "this fraction of the median magnitude",
    )
    min_rate_deviation_px_s: float = Field(
        default=3.0, description="Absolute deviation floor (px/s) so slow chains aren't over-gated"
    )


class ExposureTimeConfig(BaseModel):
    """Configuration for exposure time header keys."""

    exposure_time_keys: list[str] = Field(default_factory=list, description="FITS header keys for exposure time")


class ObservationTimeConfig(BaseModel):
    """Configuration for observation time header keys."""

    observation_time_keys: list[str] = Field(default_factory=list, description="FITS header keys for observation time")
    format: str = Field(
        default="iso",
        description="Time format (supported: 'iso', or use datetime format code '%Y-%m-%dT%H:%M:%S.%f' or similar)",
    )


class SiteConfig(BaseModel):
    """Configuration for observatory site header keys."""

    site_latitude_keys: list[str] = Field(default_factory=list, description="FITS header keys for site latitude")
    site_longitude_keys: list[str] = Field(default_factory=list, description="FITS header keys for site longitude")
    site_altitude_keys: list[str] = Field(default_factory=list, description="FITS header keys for site altitude")
    positional_format: str = Field(
        default="sexagesimal",
        description="Format for positional values (supported: 'sexagesimal', 'float')",
    )
    positional_unit: str = Field(default="degrees", description="Unit for positional values")
    altitude_unit: str = Field(default="kilometers", description="Unit for altitude")


class PointingConfig(BaseModel):
    """Configuration for telescope pointing header keys."""

    boresight_azimuth_keys: list[str] = Field(
        default_factory=list, description="FITS header keys for boresight azimuth"
    )
    boresight_altitude_keys: list[str] = Field(
        default_factory=list, description="FITS header keys for boresight altitude"
    )
    ra_dec_format: str = Field(
        default="sexagesimal",
        description="Format for RA and DEC (supported: 'sexagesimal', 'float')",
    )
    ra_units: str = Field(default="hours", description="Unit for RA (supported: 'hours', 'degrees')")
    dec_units: str = Field(default="degrees", description="Unit for DEC")
    target_ra_keys: list[str] = Field(default_factory=list, description="FITS header keys for target RA")
    target_dec_keys: list[str] = Field(default_factory=list, description="FITS header keys for target DEC")


class TrackingConfig(BaseModel):
    """Configuration for telescope tracking header keys."""

    track_ra_rate_keys: list[str] = Field(default_factory=list, description="FITS header keys for RA tracking rate")
    track_dec_rate_keys: list[str] = Field(default_factory=list, description="FITS header keys for DEC tracking rate")
    track_ra_rate_unit: str = Field(default="arcseconds/second", description="Unit for RA tracking rate")
    track_dec_rate_unit: str = Field(default="arcseconds/second", description="Unit for DEC tracking rate")
    track_mode_keys: list[str] = Field(default_factory=list, description="FITS header keys for tracking mode")
    data_fallback_enabled: bool = Field(
        default=True,
        description="When a frame has no usable TRKMODE, look at the pixels (round "
        "sources -> sidereal, streaked+aligned -> rate) to settle sidereal-vs-rate. "
        "Only runs when the header can't decide, so it costs nothing on metadata-tagged "
        "frames; set false to fall back to rate magnitude alone.",
    )
    sidereal_rate_threshold_arcsec_per_second: float = Field(
        default=1.0,
        description="When classifying by RA/DEC rate magnitude (no TRKMODE), |rate| at or "
        "below this is treated as sidereal, above as rate.",
    )


class HeadersConfig(BaseModel):
    """Configuration for FITS header mappings."""

    exposure_time: ExposureTimeConfig = Field(
        default_factory=ExposureTimeConfig,
        description="Exposure time header configuration",
    )
    observation_time: ObservationTimeConfig = Field(
        default_factory=ObservationTimeConfig,
        description="Observation time header configuration",
    )
    site: SiteConfig = Field(default_factory=SiteConfig, description="Site header configuration")
    pointing: PointingConfig = Field(default_factory=PointingConfig, description="Pointing header configuration")
    tracking: TrackingConfig = Field(default_factory=TrackingConfig, description="Tracking header configuration")
    filter_keys: list[str] = Field(
        default=["FILTER", "FILTER1", "INSFILTE"],
        description="FITS header keys for observation filter",
    )


class PhotometryConfig(BaseModel):
    """Configuration for photometry measurements.

    Single source of truth for all photometry knobs — the engine
    (senpai.engine.photometry.utils) uses this class directly.
    """

    enable: bool = Field(
        default=True,
        description="Run the per-frame photometry stage (zero point, limiting magnitude, "
        "detection photometry) after detection. Disable for detection-only runs.",
    )

    # Aperture photometry: fixed aperture size as multiple of FWHM
    aperture_radius_factor: float = Field(default=2.0, description="Aperture radius as multiple of FWHM")

    # Background annulus
    bg_inner_factor: float = Field(default=3.0, description="Background inner radius as multiple of FWHM")
    bg_outer_factor: float = Field(default=5.0, description="Background outer radius as multiple of FWHM")

    # Quality thresholds
    min_snr: float = Field(default=3.0, description="Minimum signal-to-noise ratio for the quality flag")
    max_crowding: float = Field(default=0.3, description="Maximum crowding factor for the quality flag")

    # Crowding / blending control for calibration stars. Used when selecting
    # stars for zero point and limiting magnitude, to avoid blended sources.
    isolation_radius_factor: float = Field(
        default=2.0, description="Isolation radius in units of photometric aperture radius"
    )
    isolation_delta_mag: float = Field(
        default=2.0, description="Minimum magnitude difference for a 'much brighter' neighbor"
    )

    # Limiting magnitude estimation
    limiting_snr: float = Field(
        default=3.0,
        description="SNR threshold used when estimating limiting magnitude (e.g., 3 or 5).",
    )
    limiting_completeness_fraction: float = Field(
        default=0.5,
        description="Completeness fraction for limiting magnitude (e.g., 0.5 for 50% of catalog stars above limiting_snr).",
    )
    completeness_isolate: bool = Field(
        default=True,
        description="Drop catalog stars blended with brighter neighbors from the completeness curve",
    )

    # Zero-point star selection. The ZP must come from well-measured stars only:
    # a faint catalog tail (where forced photometry latches onto neighbour flux /
    # trails and reports a spurious SNR floor) biases the median ZP up by ~1 mag.
    zp_min_snr: float = Field(default=20.0, description="Only stars at/above this SNR contribute to the zero point")
    zp_max_crowding: float = Field(default=0.2, description="...and below this crowding factor")
    zp_sigma_clip: float = Field(default=3.0, description="Sigma-clip threshold on the per-star ZP values")
    zp_min_stars: int = Field(default=8, description="Need at least this many stars to trust the high-SNR cut")

    # Uncertainty estimation
    include_read_noise: bool = Field(default=True, description="Include read noise in uncertainty")
    read_noise: float = Field(default=5.0, description="Read noise in electrons")
    gain: float = Field(default=1.0, description="Gain in electrons per ADU")

    # Magnitude selection for open band observations
    preferred_filters: list[str] = Field(
        default=["Johnson_V", "Johnson_R", "Sloan_r", "Gaia_G", "Sloan_g", "Johnson_B"],
        description="Preferred filters in order of preference",
    )

    # Multi-band calibration
    target_bands: list[str] = Field(
        default=["Johnson_V", "Sloan_r", "Gaia_G"],
        description="Target photometric bands for multi-band zero point calibration",
    )
    color_index_bands: tuple[str, str] = Field(
        default=("Gaia_BP", "Gaia_RP"),
        description="Bands forming the color index for color-term corrections",
    )
    enable_color_terms: bool = Field(
        default=True,
        description="Enable color term corrections in multi-band calibration",
    )


class CalibrationsConfig(BaseModel):
    """Configuration for calibration frames (flats, darks, etc.)."""

    master_flats_dir: str | None = Field(default=None, description="Directory containing master flat files")
    master_darks_dir: str | None = Field(default=None, description="Directory containing master dark files")
    auto_apply_flats: bool = Field(
        default=False,
        description="Automatically apply master flats during preprocessing",
    )
    auto_apply_darks: bool = Field(
        default=False,
        description="Automatically apply master darks during preprocessing",
    )
    dark_matching_headers: list[str] = Field(
        default=["XBINNING", "EXPTIME"],
        description="FITS header keywords that must match for dark frames (exposure time is handled separately)",
    )
    flat_matching_headers: list[str] = Field(
        default=["XBINNING", "FILTER"],
        description="FITS header keywords that must match for flat frames",
    )
    max_dark_exposure_ratio: float = Field(
        default=2.0,
        description="Maximum ratio between image and dark exposure times for automatic scaling",
    )

    # Preprocessing steps configuration
    auto_remove_row_median: bool = Field(
        default=True,
        description="Automatically remove row medians during preprocessing",
    )
    auto_remove_column_median: bool = Field(
        default=True,
        description="Automatically remove column medians during preprocessing",
    )
    auto_subtract_background: bool = Field(
        default=True,
        description="Automatically subtract background during preprocessing",
    )

    # Background subtraction parameters
    background_box_size: int = Field(default=20, description="Box size for background estimation")
    background_filter_size: int = Field(default=3, description="Filter size for background estimation")
    background_exclude_percentile: float = Field(
        default=50.0, description="Percentile to exclude in background estimation"
    )
    background_sigma: float = Field(default=3.0, description="Sigma for background estimation")
    background_maxiters: int = Field(default=10, description="Maximum iterations for background estimation")

    # Image scaling configuration
    auto_scale_images: bool = Field(default=False, description="Automatically scale images to optimize FWHM")
    scaling_method: str = Field(
        default="block_median",
        description="Scaling method: 'block_median' (fast + hot pixel removal) or 'blur_decimate' (better photometry)",
    )
    target_fwhm: float = Field(default=3.0, description="Target FWHM in pixels after scaling")
    oversample_threshold: float = Field(default=4.0, description="Only scale images if FWHM > this threshold")


class ObservationsConfig(BaseModel):
    """Observation-pointing configuration for the detection-pointing core.

    Consumed by :mod:`senpai.engine.pointing` when deriving per-detection RA/Dec
    and uncertainties from the refined WCS.
    """

    centroid_localization_std_pix: float | None = Field(
        default=None,
        description=(
            "1-sigma centroid-localization uncertainty, in pixels. Drives the per-observation "
            "WCS-Jacobian uncertainty derivation. Left unset (null), the derived RA/Dec "
            "uncertainties are None."
        ),
    )
    uncertainty_warn_threshold_deg: float = Field(
        default=1.0,
        description=(
            "Emit a warning when a WCS-derived RA/Dec uncertainty exceeds this threshold "
            "(degrees). Observability only; the value is still emitted."
        ),
    )
    time_offsets_s: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-sensor clock offset, in seconds, SUBTRACTED from the reported exposure window. "
            "Keyed by sensor id."
        ),
    )


class AppConfig(BaseSettings):
    """Application configuration.

    Loaded from a YAML file via ``initialize_config`` with environment-variable
    overrides (``__`` nesting delimiter, e.g. ``DETECTION__SNR_THRESHOLD=5``).
    Environment variables take precedence over YAML values. The top level is
    frozen; runtime-mutable sections (``runtime``, ``plotting``) stay mutable.
    """

    version: str = Field(description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    astrometry: AstrometryConfig = Field(default_factory=AstrometryConfig, description="Astrometry configuration")
    star_catalog: StarCatalogConfig = Field(default_factory=StarCatalogConfig, description="Star catalog configuration")
    plotting: PlottingConfig = Field(default_factory=PlottingConfig, description="Plotting configuration")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig, description="Runtime configuration options")
    detection: DetectionConfig = Field(default_factory=DetectionConfig, description="Detection configuration")
    streak: StreakDetectionConfig = Field(
        default_factory=StreakDetectionConfig,
        description="Streak detection and tracking configuration",
    )
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation configuration")
    wcs_validation: WCSValidationConfig = Field(
        default_factory=WCSValidationConfig,
        description="Absolute post-refinement WCS validation configuration",
    )
    chain_gate: ChainGateConfig = Field(
        default_factory=ChainGateConfig,
        description="Frame-shift chain consistency gate configuration",
    )
    headers: HeadersConfig = Field(default_factory=HeadersConfig, description="FITS header mapping configuration")
    photometry: PhotometryConfig = Field(default_factory=PhotometryConfig, description="Photometry configuration")
    calibrations: CalibrationsConfig = Field(
        default_factory=CalibrationsConfig,
        description="Calibration frames configuration",
    )
    observations: ObservationsConfig = Field(
        default_factory=ObservationsConfig,
        description="Observation-pointing configuration",
    )

    model_config = SettingsConfigDict(env_nested_delimiter="__", frozen=True, extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the settings sources: init kwargs, then env vars, then the YAML file.

        Args:
            settings_cls (type[BaseSettings]): the settings class being configured.
            init_settings (PydanticBaseSettingsSource): source for values passed at init.
            env_settings (PydanticBaseSettingsSource): source for environment variables.
            dotenv_settings (PydanticBaseSettingsSource): source for ``.env`` files (unused).
            file_secret_settings (PydanticBaseSettingsSource): file-secret source (unused).

        Returns:
            tuple[PydanticBaseSettingsSource, ...]: the ordered sources; earlier sources
                take precedence, so environment variables override the YAML file.
        """
        return (
            init_settings,
            env_settings,
            _YamlConfigSource(settings_cls, yaml_file=_yaml_file_for_next_init),
        )


def get_config() -> AppConfig:
    """Get the global config instance.

    Returns:
        AppConfig: The global configuration instance.

    Raises:
        RuntimeError: if the config has not been initialized yet.
    """
    global _config_instance

    if _config_instance is None:
        raise RuntimeError("Config not initialized")

    return _config_instance


def initialize_config(config_path: Path) -> AppConfig:
    """Initialize the global config instance from a YAML file.

    The file may be flat (sections at the top level) or nested under a single
    ``app:`` key. Environment variables (``__`` nesting delimiter) override the
    file's values.

    Args:
        config_path (Path): Path to the config YAML.

    Returns:
        AppConfig: Configuration instance.

    Raises:
        Exception: whatever pydantic raises when the merged configuration is invalid.
    """
    global _config_instance, _yaml_file_for_next_init

    logger.info(f"Loading configuration from {config_path}")
    _yaml_file_for_next_init = Path(config_path)
    try:
        config = AppConfig()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    finally:
        # Direct AppConfig(**kwargs) constructions must not re-read this file.
        _yaml_file_for_next_init = None

    _config_instance = config
    return config


def get_or_initialize_config(config_path: Path | None = None) -> AppConfig:
    """Get the loaded config; if none, load ``config_path``, ``SENPAI_CONFIG_PATH``, or the local override.

    Args:
        config_path (Path | None): explicit config path to load when uninitialized.

    Returns:
        AppConfig: Configuration instance.
    """
    try:
        config = get_config()
    except RuntimeError:
        if config_path:
            config = initialize_config(config_path)
        elif os.environ.get("SENPAI_CONFIG_PATH"):
            env_path = Path(os.environ["SENPAI_CONFIG_PATH"])
            logger.info(f"No config initialized, using SENPAI_CONFIG_PATH={env_path}")
            config = initialize_config(env_path)
        else:
            from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE

            # The local override lives in the source tree's resources/config and is NOT
            # packaged in the wheel. In an installed context (e.g. a downstream v3 thin
            # wrapper importing astro-senpai) it is absent, and loading it would yield a
            # confusing multi-field pydantic validation cascade. Fail with a clear,
            # actionable message instead: callers must supply a config.
            if not LOCAL_APP_CONFIG_OVERRIDE.exists():
                raise ConfigError(
                    "No SENPAI configuration available: pass a config path, set the "
                    "SENPAI_CONFIG_PATH environment variable, or (for a source checkout) "
                    f"provide {LOCAL_APP_CONFIG_OVERRIDE}."
                ) from None
            logger.info(f"No config initialized, using {LOCAL_APP_CONFIG_OVERRIDE}")
            config = initialize_config(LOCAL_APP_CONFIG_OVERRIDE)

    return config
