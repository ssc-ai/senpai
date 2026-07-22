"""Burr observatory integration.

Reads ``run_state.json`` plus per-sensor FITS directories into frame batches
that the senpai collect pipeline can consume.
"""

from senpai.integrations.burr.filenames import (
    ParsedFilename,
    parse_burr_filename,
)
from senpai.integrations.burr.night import (
    BurrNight,
    FrameBatch,
    FrameRecord,
)
from senpai.integrations.burr.run_state import (
    ExecutedCommand,
    RunState,
)

__all__ = [
    "BurrNight",
    "ExecutedCommand",
    "FrameBatch",
    "FrameRecord",
    "ParsedFilename",
    "RunState",
    "parse_burr_filename",
]
