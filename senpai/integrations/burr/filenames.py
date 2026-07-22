"""Filename parsers for burr FITS files.

Two naming conventions live in the data tree:

* Semantic (Hornet, newer controller1):
      ``YYYYMMDDTHHMMSS_<task>_<target_id>_f<N>.fits``
  e.g. ``20260527T071650_calsats_41175_f0.fits``,
       ``20260527T072037_coverage_AltAzTarget_f0.fits``,
       ``20260428T053218_twilight_flats_AltAzTarget_f0.fits``.

* Opaque UUID (older controller1):
      ``<uuid>.fits``
  e.g. ``0073b353-3f9f-11f1-9659-010101010000.fits``.
  These can only be linked back to the run_state command log by reading the
  FITS header's DATE-OBS and matching by timestamp; that resolution lives in
  :mod:`night`, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Task strings in filenames → command verbs in run_state.executed_commands.
TASK_TO_COMMAND: dict[str, str] = {
    "calsats": "calsat_observed",
    "coverage": "coverage_point_observed",
    "photometric_standards": "photometric_standards_observed",
    "twilight_flats": "flat_field_saved",
    "lunar_background": "lunar_background_observed",
}

# The semantic name is ``<ts>_<task>_<target>_f<idx>``, but both the task and the
# target can contain underscores (``photometric_standards``; targets like
# ``SAT_26605``, ``104_485``, ``BD_+5_2468``). The only unambiguous way to split
# task from target is to anchor on the known task tokens — longest first so
# multi-word tasks win over any (currently non-existent) prefix collision — and
# let the target be everything up to the trailing ``_f<idx>``. An unknown task
# token falls through to the "unrecognized" record so the caller can fall back to
# header inspection rather than mis-splitting it.
_TASK_ALT = "|".join(sorted(TASK_TO_COMMAND, key=len, reverse=True))
_SEMANTIC_RE = re.compile(
    rf"^(?P<ts>\d{{8}}T\d{{6}})_(?P<task>{_TASK_ALT})_(?P<target>.+)_f(?P<idx>\d+)$"
)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


@dataclass(frozen=True, slots=True)
class ParsedFilename:
    """Structured fields parsed from a burr FITS filename.

    Attributes:
        path: The source file path.
        timestamp: Capture time (tz-aware UTC), or None for UUID-style names.
        task: Task token (e.g. ``"calsats"``), or None for UUID-style names.
        target: NORAD id or target name, or None for UUID-style names.
        frame_index: 0-based frame index, or None for UUID-style names.
        is_uuid: True when the filename is an opaque UUID-style name.
    """

    path: Path
    timestamp: datetime | None  # tz-aware UTC, or None for UUID-style names
    task: str | None  # e.g. "calsats"; None for UUID-style
    target: str | None  # NORAD id or AltAzTarget/RateTarget/ICRSTarget; None for UUID-style
    frame_index: int | None  # 0-based; None for UUID-style
    is_uuid: bool

    @property
    def command_verb(self) -> str | None:
        """The run_state command verb for this file's task, or None if unknown."""
        return TASK_TO_COMMAND.get(self.task) if self.task else None


def parse_burr_filename(path: str | Path) -> ParsedFilename:
    """Parse a burr FITS filename into structured fields.

    Returns a :class:`ParsedFilename` with ``is_uuid=True`` and everything else
    None for opaque names. Never raises — unknown naming patterns return a
    record where every parsed field is None and is_uuid is False, so the caller
    can decide whether to skip or fall back to header inspection.
    """
    p = Path(path)
    stem = p.stem

    if m := _SEMANTIC_RE.match(stem):
        ts = datetime.strptime(m["ts"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        return ParsedFilename(
            path=p,
            timestamp=ts,
            task=m["task"],
            target=m["target"],
            frame_index=int(m["idx"]),
            is_uuid=False,
        )

    if _UUID_RE.match(stem):
        return ParsedFilename(
            path=p, timestamp=None, task=None, target=None, frame_index=None, is_uuid=True
        )

    return ParsedFilename(
        path=p, timestamp=None, task=None, target=None, frame_index=None, is_uuid=False
    )
