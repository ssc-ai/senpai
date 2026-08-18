"""Turn a burr night's outputs into batches the collect pipeline can consume.

Reads ``run_state.json`` alongside the per-sensor FITS directories.
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
