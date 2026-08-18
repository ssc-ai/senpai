"""Aggregate a night's per-frame photometry into per-night calibration products.

Zero point, extinction, limiting magnitude and Az/Alt coverage, from the photometry
summaries of the night's batches.

This package replaces the previous monolithic analyzer.py (and the three plot
sibling files) with a slim post-stage that consumes ``SenpaiRun`` JSONs
written by :mod:`senpai.cli.burr` rather than re-doing astrometry + photometry.
"""

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
