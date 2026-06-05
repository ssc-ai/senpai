"""BurrNight: glue between a per-night run_state.json + a sensor's flat FITS dir.

A ``BurrNight`` knows where to find both the metadata sidecar and the raw FITS
for one sensor on one night, can index every frame in that night, and can
group those frames into ``FrameBatch`` objects — one per collection event the
burr controller logged. Each batch is what you hand to
``senpai.engine.processing.collect.process_senpai_collect`` to produce a
``SenpaiRun``.

Layout assumption (matching the deployed burr tree)::

    <burr_root>/
        <sensor>/                       # flat FITS dir, e.g. Hornet/, controller1/
        burr/<sensor>_<YYYYMMDD>/metadata/run_state.json

The sensor directory is shared across nights; a night is delimited by the date
in the metadata directory name, which we apply as a [start, end) window on
filename timestamps.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from senpai.engine.utils.frame_organization import extract_id_from_header
from senpai.integrations.burr.filenames import ParsedFilename, parse_burr_filename
from senpai.integrations.burr.run_state import ExecutedCommand, RunState

logger = logging.getLogger(__name__)

# Window (seconds) within which a frame's filename timestamp must fall after a
# command's observation_time to be attributed to it. Coverage points run 3
# sequential sub-exposures (sidereal + rate + rate) plus readouts; in practice
# the trailing frames land 150–200s after the command logs the start. 300s is
# the safe upper bound — the "latest preceding command of same type" rule in
# :func:`_attribute_command` prevents overlap into the next collection.
_DEFAULT_ATTRIBUTION_WINDOW_S: float = 300.0

# Target tokens in semantic filenames that pin tracking mode regardless of the
# frame-index position in the command's tracking_modes list. Used for coverage
# and photometric_standards, where ``_fN`` is intra-exposure sub-frame index
# (rate exposures are split into 2 halves: f0 + f1) rather than the index into
# the per-collection tracking_modes vector.
_SIDEREAL_TARGET_TOKENS: frozenset[str] = frozenset({"AltAzTarget", "ICRSTarget"})
_RATE_TARGET_TOKENS: frozenset[str] = frozenset({"RateTarget"})

# Inter-frame gap (seconds) above which we treat orphan frames of the same task
# as belonging to *different* pointings. Within-pointing gaps for burr's
# photometric_standards and twilight_flats sequences are 7–25s; the next
# pointing's first frame is typically ≥60s later (intervening slew + a
# logged collection in between).
_ORPHAN_CLUSTER_GAP_S: float = 60.0


def _utc_offset(run_state: RunState) -> timedelta:
    """Best-effort UTC offset for the site, used to label observing nights by
    their evening-local date the way burr does. Prefer the tz baked into
    ``observation_date``; fall back to longitude/15; default to UTC."""

    od = run_state.observation_date
    if od:
        try:
            dt = datetime.fromisoformat(od)
        except ValueError:
            dt = None
        if dt is not None and dt.utcoffset() is not None:
            return dt.utcoffset()
    site = run_state.config.site
    if site and site.longitude is not None:
        return timedelta(hours=round(site.longitude / 15.0))
    return timedelta(0)


def _tracking_mode_from_target(target: str | None) -> str | None:
    if target is None:
        return None
    if target in _SIDEREAL_TARGET_TOKENS:
        return "sidereal"
    if target in _RATE_TARGET_TOKENS:
        return "rate"
    return None

# Night-id pattern: <Sensor>_YYYYMMDD (sensor may contain hyphens, e.g. DAO-01)
_NIGHT_DIR_RE = re.compile(r"^(?P<sensor>[A-Za-z][A-Za-z0-9-]*)_(?P<date>\d{8})$")


@dataclass(slots=True)
class FrameRecord:
    """One FITS frame located in the sensor data dir, enriched with what we
    could learn from the filename + run_state command log."""

    path: Path
    parsed: ParsedFilename
    command: ExecutedCommand | None = None
    # Tracking mode the burr controller intended for this frame, sourced from
    # ``command.tracking_modes[frame_index]`` when both are available. The
    # senpai collect pipeline will re-derive this from the FITS header — this
    # field is a hint, not authoritative.
    intended_tracking_mode: str | None = None
    # Logical-set id read from a FITS header keyword (e.g. BURRSEQ) when frames
    # are batched by header rather than filename heuristics. None unless a
    # ``seq_key`` was supplied to :meth:`BurrNight.index_frames`.
    seq_id: str | None = None

    @property
    def timestamp(self) -> datetime | None:
        return self.parsed.timestamp

    @property
    def task(self) -> str | None:
        return self.parsed.task


@dataclass(slots=True)
class FrameBatch:
    """A group of frames that belong to one collection event. Hand the
    ``paths`` to senpai's collect pipeline as a single SenpaiRun."""

    batch_id: str
    task: str | None  # "calsats", "coverage", "photometric_standards", ...
    command: ExecutedCommand | None
    frames: list[FrameRecord] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        return [f.path for f in self.frames]

    @property
    def has_intended_rate_frames(self) -> bool:
        return any(f.intended_tracking_mode == "rate" for f in self.frames)


class BurrNight(BaseModel):
    """One night of burr data for one sensor."""

    model_config = {"arbitrary_types_allowed": True}

    burr_root: Path
    night_id: str
    sensor: str
    date_str: str  # YYYYMMDD as written in the night dir name
    metadata_dir: Path
    run_state_path: Path
    data_dir: Path
    run_state: RunState
    # Explicit [start, end) UTC window override. Set by :meth:`auto_nights` when
    # a flat data dir holds several observing nights that the run_state's single
    # lighting_schedule can't delimit; None falls back to the schedule.
    window_start: datetime | None = None
    window_end: datetime | None = None

    # --- construction ---------------------------------------------------------

    @classmethod
    def from_night_dir(
        cls,
        night_dir: str | Path,
        burr_root: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> "BurrNight":
        """Build a BurrNight from a path like ``/burr/burr/Hornet_20260527``.

        ``burr_root`` defaults to ``night_dir.parent.parent`` (the burr tree root
        that owns both ``<sensor>/`` and ``burr/<night_id>/``).  ``data_dir``
        defaults to ``<burr_root>/<sensor>``.
        """

        night_dir = Path(night_dir).resolve()
        m = _NIGHT_DIR_RE.match(night_dir.name)
        if not m:
            raise ValueError(
                f"Night directory name {night_dir.name!r} does not match "
                "<Sensor>_YYYYMMDD"
            )
        sensor = m["sensor"]
        date_str = m["date"]

        if burr_root is None:
            burr_root = night_dir.parent.parent
        burr_root = Path(burr_root).resolve()

        metadata_dir = night_dir / "metadata"
        run_state_path = metadata_dir / "run_state.json"
        if not run_state_path.is_file():
            raise FileNotFoundError(f"run_state.json not found at {run_state_path}")

        if data_dir is None:
            data_dir = burr_root / sensor
        data_dir = Path(data_dir).resolve()

        return cls(
            burr_root=burr_root,
            night_id=night_dir.name,
            sensor=sensor,
            date_str=date_str,
            metadata_dir=metadata_dir,
            run_state_path=run_state_path,
            data_dir=data_dir,
            run_state=RunState.load(run_state_path),
        )

    @classmethod
    def auto_nights(
        cls,
        run_state_path: str | Path,
        data_dir: str | Path,
        *,
        sensor: str | None = None,
        gap_hours: float = 3.0,
        pad_seconds: float = 120.0,
    ) -> list["BurrNight"]:
        """Split a flat FITS dir into one :class:`BurrNight` per observing night.

        Burr is supposed to write one night per directory + run_state, but a
        controller bug can dump several nights' frames into a single flat dir
        against a single (stale) run_state. This recovers the per-night split
        the controller should have produced: frames are grouped by gaps larger
        than ``gap_hours`` in their filename timestamps, each group becomes a
        night with an explicit frame-derived window and a
        ``<sensor>_<YYYYMMDD>`` id (evening-local date, matching burr's
        convention). The run_state is reused only for site config — its command
        log and lighting window describe a different night and are ignored, so
        every frame batches via the command-less ``(task, target)`` path.
        """

        run_state_path = Path(run_state_path).resolve()
        data_dir = Path(data_dir).resolve()
        run_state = RunState.load(run_state_path)

        if sensor is None:
            site = run_state.config.site
            sensor = site.name if site and site.name else data_dir.name

        offset = _utc_offset(run_state)

        timestamps: list[datetime] = []
        for path in sorted(data_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".fits", ".fit", ".fts"):
                continue
            parsed = parse_burr_filename(path)
            if parsed.timestamp is not None:
                timestamps.append(parsed.timestamp)

        if not timestamps:
            logger.warning("auto_nights: no timestamped frames in %s", data_dir)
            return []

        timestamps.sort()
        gap = timedelta(hours=gap_hours)
        groups: list[list[datetime]] = [[timestamps[0]]]
        for ts in timestamps[1:]:
            if ts - groups[-1][-1] > gap:
                groups.append([ts])
            else:
                groups[-1].append(ts)

        pad = timedelta(seconds=pad_seconds)
        nights: list[BurrNight] = []
        for group in groups:
            date_str = (group[0] + offset - timedelta(hours=12)).strftime("%Y%m%d")
            nights.append(
                cls(
                    burr_root=data_dir.parent,
                    night_id=f"{sensor}_{date_str}",
                    sensor=sensor,
                    date_str=date_str,
                    metadata_dir=run_state_path.parent,
                    run_state_path=run_state_path,
                    data_dir=data_dir,
                    run_state=run_state,
                    window_start=group[0] - pad,
                    window_end=group[-1] + pad,
                )
            )
        logger.info(
            "auto_nights: %s → %d night(s): %s",
            data_dir, len(nights), ", ".join(n.night_id for n in nights),
        )
        return nights

    # --- night window ---------------------------------------------------------

    def night_window(self) -> tuple[datetime, datetime]:
        """[start, end) UTC window the night covers.

        An explicit ``window_start``/``window_end`` override wins (set by
        :meth:`auto_nights`). Otherwise prefer the lighting_schedule
        (night_start/morning_civil_end) when present; failing that fall back to
        a ±18h window around local midnight of the date in the night-dir name.
        """

        if self.window_start is not None and self.window_end is not None:
            return self.window_start, self.window_end

        ls = self.run_state.lighting_schedule
        if ls and ls.night_start and ls.night_end:
            start = datetime.fromisoformat(ls.night_start)
            end = datetime.fromisoformat(ls.night_end) + timedelta(hours=2)
            return start, end

        # Fallback: dawn-to-dawn UTC window around the burr-named date.
        d = datetime.strptime(self.date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        return d - timedelta(hours=6), d + timedelta(hours=30)

    # --- frame indexing -------------------------------------------------------

    def index_frames(
        self,
        attribution_window_s: float = _DEFAULT_ATTRIBUTION_WINDOW_S,
        seq_key: str | None = None,
    ) -> list[FrameRecord]:
        """Scan the sensor data dir for FITS files, keep those whose filename
        timestamp lands in the night window, and attribute each to a logged
        command when possible. UUID-style filenames are kept but not attributed
        here (header-based resolution is a separate, deferred concern).

        When ``seq_key`` is given, each kept frame's ``seq_id`` is read from that
        FITS header keyword (via the shared
        :func:`~senpai.engine.utils.frame_organization.extract_id_from_header`),
        so the batcher can group on the controller's own logical-set id (e.g.
        ``BURRSEQ``) instead of filename heuristics. This opens each frame's
        header — cheap relative to processing, but not free on a full night.
        """

        if not self.data_dir.is_dir():
            logger.warning("burr data dir does not exist: %s", self.data_dir)
            return []

        start, end = self.night_window()
        commands = self.run_state.collection_commands()

        records: list[FrameRecord] = []
        kept = 0
        skipped_out_of_window = 0
        skipped_unrecognized = 0
        uuid_count = 0
        for path in sorted(self.data_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".fits", ".fit", ".fts"):
                continue

            parsed = parse_burr_filename(path)

            if parsed.is_uuid:
                uuid_count += 1
                records.append(FrameRecord(path=path, parsed=parsed))
                continue

            if parsed.timestamp is None:
                skipped_unrecognized += 1
                continue

            if not (start <= parsed.timestamp < end):
                skipped_out_of_window += 1
                continue

            command = _attribute_command(parsed, commands, attribution_window_s)

            # Prefer the target-token mapping when it's defined (coverage,
            # photometric_standards). Otherwise (calsats, whose target is a
            # NORAD id) fall back to the command's tracking_modes vector
            # indexed by frame_index.
            intended_mode = _tracking_mode_from_target(parsed.target)
            if (
                intended_mode is None
                and command is not None
                and parsed.frame_index is not None
            ):
                modes = command.tracking_modes
                if 0 <= parsed.frame_index < len(modes):
                    intended_mode = modes[parsed.frame_index]

            records.append(
                FrameRecord(
                    path=path,
                    parsed=parsed,
                    command=command,
                    intended_tracking_mode=intended_mode,
                )
            )
            kept += 1

        if seq_key:
            n_missing = 0
            for r in records:
                try:
                    r.seq_id = extract_id_from_header(r.path, seq_key)
                except Exception as e:  # unreadable header — keep the frame, no seq
                    logger.warning("seq_key %s unreadable for %s: %s", seq_key, r.path.name, e)
                    r.seq_id = None
                if r.seq_id is None:
                    n_missing += 1
            if n_missing:
                logger.warning(
                    "BurrNight %s: %d/%d frames missing seq_key %s "
                    "(they fall back to command/orphan batching)",
                    self.night_id, n_missing, len(records), seq_key,
                )

        logger.info(
            "BurrNight %s: indexed %d frames (%d in-window semantic + %d uuid), "
            "skipped %d out-of-window, %d unrecognized",
            self.night_id, len(records), kept, uuid_count,
            skipped_out_of_window, skipped_unrecognized,
        )
        return records

    # --- batching -------------------------------------------------------------

    def frame_batches(
        self,
        attribution_window_s: float = _DEFAULT_ATTRIBUTION_WINDOW_S,
        seq_key: str | None = None,
    ) -> Iterator[FrameBatch]:
        """Yield one FrameBatch per collection event.

        With ``seq_key`` (e.g. ``"BURRSEQ"``) frames are grouped by that FITS
        header id — the controller's own logical-set marker — which is the
        authoritative split for rate-sidereal work (one sidereal anchor + its
        rate sub-frames per set). This is preferred for burr data that carries
        the keyword. Frames missing the keyword fall back to the command/orphan
        batching below.

        Without ``seq_key`` (or for frames missing it), batches are formed from
        the run_state command log where available, and otherwise by clustering
        orphan frames per ``(task, pointing)`` and time proximity.
        """

        records = self.index_frames(attribution_window_s, seq_key=seq_key)

        if seq_key:
            with_seq = [r for r in records if r.seq_id]
            without_seq = [r for r in records if not r.seq_id]
            yield from self._emit_seq_batches(with_seq)
            yield from self._emit_command_and_orphan_batches(without_seq)
            return

        yield from self._emit_command_and_orphan_batches(records)

    def _emit_seq_batches(self, records: list[FrameRecord]) -> Iterator[FrameBatch]:
        """One batch per distinct ``seq_id`` (e.g. BURRSEQ), ordered by the
        set's earliest frame. Each set is exactly the controller's logical
        collection unit — for coverage/photometric that is a single exposure's
        sidereal anchor plus its rate sub-frames; for calsats the full sequence."""

        by_seq: dict[str, list[FrameRecord]] = {}
        for r in records:
            by_seq.setdefault(r.seq_id, []).append(r)

        def _min_ts(rs: list[FrameRecord]) -> datetime:
            stamps = [r.parsed.timestamp for r in rs if r.parsed.timestamp]
            return min(stamps) if stamps else datetime.min.replace(tzinfo=timezone.utc)

        for seq_id, rs in sorted(by_seq.items(), key=lambda kv: _min_ts(kv[1])):
            rs.sort(key=lambda r: (r.parsed.timestamp or datetime.min.replace(tzinfo=timezone.utc), r.parsed.frame_index or 0, r.path.name))
            _infer_intended_modes(rs)
            head = rs[0]
            ts_tag = _min_ts(rs).strftime("%Y%m%dT%H%M%S")
            target = head.parsed.target or "unknown"
            cmd = next((r.command for r in rs if r.command is not None), None)
            yield FrameBatch(
                batch_id=f"{self.night_id}_{ts_tag}_{head.parsed.task or 'unknown'}_{target}_{str(seq_id)[:8]}",
                task=head.parsed.task,
                command=cmd,
                frames=rs,
            )

    def _emit_command_and_orphan_batches(
        self, records: list[FrameRecord]
    ) -> Iterator[FrameBatch]:
        """Batch frames from the run_state command log where available, and
        otherwise by clustering orphan frames per ``(task, pointing)`` and time
        proximity. UUID-named / unparseable frames emit as singletons so they
        aren't dropped."""

        # First pass: group by command identity.
        by_command: dict[int, list[FrameRecord]] = {}
        orphans: list[FrameRecord] = []
        for r in records:
            if r.command is None:
                orphans.append(r)
            else:
                by_command.setdefault(id(r.command), []).append(r)

        # Emit attributed batches in command-time order.
        attributed = sorted(
            by_command.values(),
            key=lambda rs: rs[0].command.observation_time or datetime.min.replace(tzinfo=timezone.utc),
        )
        for rs in attributed:
            rs.sort(key=lambda r: (r.parsed.frame_index or 0, r.path.name))
            cmd = rs[0].command
            assert cmd is not None
            label = cmd.target_label or "unlabeled"
            ts = cmd.observation_time
            ts_tag = ts.strftime("%Y%m%dT%H%M%S") if ts else "unknownT"
            yield FrameBatch(
                batch_id=f"{self.night_id}_{ts_tag}_{cmd.command}_{label}",
                task=rs[0].parsed.task,
                command=cmd,
                frames=rs,
            )

        # Orphans: cluster per task by time proximity. UUID-style and
        # otherwise unparseable orphans emit as singletons (no safe key).
        timed_by_task: dict[str | None, list[FrameRecord]] = {}
        singletons: list[FrameRecord] = []
        for r in orphans:
            if r.parsed.timestamp is None:
                singletons.append(r)
            else:
                timed_by_task.setdefault(r.parsed.task, []).append(r)

        for task, rs in sorted(timed_by_task.items(), key=lambda kv: kv[0] or ""):
            rs.sort(key=lambda r: (r.parsed.timestamp, r.path.name))
            for cluster in _cluster_by_time_gap(
                rs, _ORPHAN_CLUSTER_GAP_S, key=_pointing_key
            ):
                head = cluster[0]
                ts_tag = head.parsed.timestamp.strftime("%Y%m%dT%H%M%S")
                target = head.parsed.target or "unknown"
                yield FrameBatch(
                    batch_id=f"{self.night_id}_{ts_tag}_{task or 'unknown'}_{target}",
                    task=task,
                    command=None,
                    frames=cluster,
                )

        for r in sorted(singletons, key=lambda r: r.path.name):
            yield FrameBatch(
                batch_id=f"{self.night_id}_uuid_{r.path.stem}"
                if r.parsed.is_uuid
                else f"{self.night_id}_unrecognized_{r.path.stem}",
                task=None,
                command=None,
                frames=[r],
            )


# --- clustering helpers -------------------------------------------------------


def _cluster_by_time_gap(
    records: list[FrameRecord],
    gap_s: float,
    key=None,
) -> Iterator[list[FrameRecord]]:
    """Greedy walk: start a new cluster whenever the gap to the previous
    record's timestamp exceeds gap_s. Assumes records are sorted by timestamp.

    With ``key`` (a record → pointing-id callable) a cluster also breaks when
    the pointing id changes between adjacent frames — so command-less nights,
    where every frame is an orphan, still split into one batch per collection
    event (per satellite / coverage pixel / standard field) even when
    consecutive events fall within ``gap_s``. A None key on either side is
    treated as "no pointing info" and never forces a break, preserving pure
    time-gap clustering for targets that encode tracking mode rather than
    pointing identity (AltAzTarget/RateTarget)."""

    if not records:
        return
    current: list[FrameRecord] = [records[0]]
    gap = timedelta(seconds=gap_s)
    for r in records[1:]:
        prev = current[-1]
        prev_ts = prev.parsed.timestamp
        gapped = prev_ts is not None and r.parsed.timestamp - prev_ts > gap
        keyed = False
        if key is not None:
            kp, kc = key(prev), key(r)
            keyed = kp is not None and kc is not None and kp != kc
        if gapped or keyed:
            yield current
            current = [r]
        else:
            current.append(r)
    yield current


def _infer_intended_modes(frames: list[FrameRecord]) -> None:
    """Fill in ``intended_tracking_mode`` for frames the header + command log
    left ambiguous, using burr's collection conventions.

    Currently handles the calsat trailing-sidereal anchor: a calsat set is
    3 rate frames + a trailing **sidereal** frame, but burr writes
    ``TRKMODE: rate`` for all four (handoff design decision #1) and the DAO-01
    nights have no matching command log to carry the intended modes. The
    sidereal leg is the last frame (highest ``frame_index`` == FRAMENUM), so
    mark it sidereal and the rest rate. This lets senpai solve the anchor as
    point sources instead of blind-solving WCS off streak centroids.

    Only fills ``None`` — an explicit command-log / target-token mode is never
    overridden (so Hornet, which logs the modes, is unaffected). Mutates in
    place; callers pass a single collection set."""

    calsats = [f for f in frames if f.parsed.task == "calsats"]
    if len(calsats) < 2:
        return
    anchor = max(calsats, key=lambda f: (f.parsed.frame_index or 0))
    for f in calsats:
        if f.intended_tracking_mode is None:
            f.intended_tracking_mode = "sidereal" if f is anchor else "rate"


def _pointing_key(record: FrameRecord) -> str | None:
    """Pointing identity of a frame, for command-less ``(task, target)``
    batching. A target that is a tracking-mode token (AltAzTarget/RateTarget/
    ICRSTarget) names the *mode* of a sub-exposure, not the pointing, so it
    carries no pointing identity (None). A target like a NORAD id, coverage
    pixel id, or standard-field id *is* the pointing."""

    target = record.parsed.target
    if target is None or _tracking_mode_from_target(target) is not None:
        return None
    return target


# --- attribution ---------------------------------------------------------------


def _attribute_command(
    parsed: ParsedFilename,
    commands: list[ExecutedCommand],
    window_s: float,
) -> ExecutedCommand | None:
    """Find the collection command this frame most likely belongs to.

    Match criteria:

    * Command type must match the filename's task (calsats → calsat_observed,
      coverage → coverage_point_observed, ...). Filenames whose task has no
      logged command verb (e.g. photometric_standards on most nights) fall
      through to None.
    * Command's ``observation_time`` must be in the window
      ``[frame_ts - window_s, frame_ts]`` (frames are written *after* the
      command logs the start of the collection).
    * Among candidates, take the most recent — the closest preceding command.
    """

    if parsed.command_verb is None or parsed.timestamp is None:
        return None

    fts = parsed.timestamp
    window = timedelta(seconds=window_s)
    best: ExecutedCommand | None = None
    best_dt: timedelta | None = None
    for c in commands:
        if c.command != parsed.command_verb:
            continue
        cts = c.observation_time
        if cts is None:
            continue
        delta = fts - cts
        if delta < timedelta(0) or delta > window:
            continue
        if best_dt is None or delta < best_dt:
            best = c
            best_dt = delta
    return best
