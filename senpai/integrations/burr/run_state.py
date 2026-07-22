"""Pydantic models for the burr controller's per-night run_state.json.

The schema is loose (lots of free-form `metadata` payloads keyed by command type),
so we model the outer shape strictly and keep command metadata as a dict, with
typed accessors for the fields downstream code actually uses.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Commands that produced one or more FITS frames. Everything else (run started,
# catalog updates, map creation, rejected flats) is purely bookkeeping.
COLLECTION_COMMANDS: frozenset[str] = frozenset({
    "calsat_observed",
    "coverage_point_observed",
    "flat_field_saved",
    "photometric_standards_observed",  # not yet observed in logs but allowed
    "lunar_background_observed",
})


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp string, returning None for empty input.

    Args:
        ts: An ISO-8601 timestamp string, or None.

    Returns:
        The parsed ``datetime``, or None when ``ts`` is falsy.
    """
    if not ts:
        return None
    # burr writes mixed offset / no-offset strings; both parse cleanly.
    return datetime.fromisoformat(ts)


class ExecutedCommand(BaseModel):
    """One entry in run_state.executed_commands[].

    The interesting payload lives in `metadata`; its shape varies per command
    type, so we expose it as a dict plus a few typed accessors for the keys
    every collection-producing command shares.
    """

    model_config = ConfigDict(extra="ignore")

    timestamp: str
    command: str
    result: str | None = None
    error: str | None = None
    stage: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_collection(self) -> bool:
        """Whether this command produced one or more FITS frames."""
        return self.command in COLLECTION_COMMANDS

    @property
    def observation_time(self) -> datetime | None:
        """Parsed observation time from the metadata, or None if absent."""
        return _parse_iso(self.metadata.get("observation_time"))

    @property
    def tracking_modes(self) -> list[str]:
        """The per-frame tracking modes recorded in the metadata."""
        return list(self.metadata.get("tracking_modes", []))

    @property
    def exposure_time(self) -> float | None:
        """The single exposure time (seconds) from the metadata, or None."""
        v = self.metadata.get("exposure_time")
        return float(v) if v is not None else None

    @property
    def exposure_times(self) -> list[float]:
        """The per-frame exposure times (seconds) recorded in the metadata."""
        return [float(x) for x in self.metadata.get("exposure_times", [])]

    @property
    def target_label(self) -> str | None:
        """Return a human-readable target identifier for this command.

        This is the NORAD id for calsats, the pixel id for coverage, etc.

        Returns:
            The most specific target id available, or None if none is present.
        """
        md = self.metadata
        if "norad_id" in md:
            return f"norad_{md['norad_id']}"
        if "map_id" in md and "pixel_id" in md:
            return f"map{md['map_id']}_pixel{md['pixel_id']}"
        return None


class SiteConfig(BaseModel):
    """Observatory site parameters from the run_state ``config.site`` block."""

    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_km: float | None = None


class RunConfig(BaseModel):
    """The ``config`` block — site + schedule + hardware.

    Only the fields downstream code reads are modeled; everything else is
    preserved via ``extra='allow'``.
    """

    model_config = ConfigDict(extra="allow")
    site: SiteConfig | None = None
    schedule: dict[str, Any] = Field(default_factory=dict)


class LightingSchedule(BaseModel):
    """Night/twilight and moon timing for the run, from ``lighting_schedule``."""

    model_config = ConfigDict(extra="allow")
    night_start: str | None = None
    night_end: str | None = None
    moon_phase: float | None = None
    moon_rise: str | None = None
    moon_set: str | None = None
    moon_is_waxing: bool | None = None


class RunState(BaseModel):
    """Top-level ``run_state.json`` model.

    Models the fields we read and allows extras for forward compatibility with
    the burr controller's evolving schema.
    """

    model_config = ConfigDict(extra="allow")

    version: str | None = None
    run_id: str
    observation_date: str | None = None
    created_at: str | None = None
    status: str | None = None
    current_stage: str | None = None
    config: RunConfig = Field(default_factory=RunConfig)
    lighting_schedule: LightingSchedule | None = None
    executed_commands: list[ExecutedCommand] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> RunState:
        """Load and validate a ``run_state.json`` file.

        Args:
            path: Path to the ``run_state.json`` file.

        Returns:
            The parsed and validated :class:`RunState`.
        """
        text = Path(path).read_text()
        return cls.model_validate_json(text)

    def collection_commands(self) -> list[ExecutedCommand]:
        """Return only the executed commands that produced FITS frames."""
        return [c for c in self.executed_commands if c.is_collection]
