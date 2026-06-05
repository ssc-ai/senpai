"""Observability post-stage: aggregates per-frame photometry summaries from a
night's batches into per-night calibration products (zero point, extinction,
limiting magnitude, Az/Alt coverage).

This package replaces the previous monolithic analyzer.py (and the three plot
sibling files) with a slim post-stage that consumes ``SenpaiRun`` JSONs
written by :mod:`senpai.cli.burr` rather than re-doing astrometry + photometry."""

from senpai.engine.observability.calibration import (
    ExtinctionFit,
    FramePhoto,
    NightCalibration,
    ZeroPointStat,
    analyze_night,
    plot_calibration,
    save_calibration,
)

__all__ = [
    "ExtinctionFit",
    "FramePhoto",
    "NightCalibration",
    "ZeroPointStat",
    "analyze_night",
    "plot_calibration",
    "save_calibration",
]
