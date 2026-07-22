"""Tests for senpai.integrations.burr filename parsing and the BurrNight batcher.

Covers filename parsing, the run_state model, and the BurrNight
indexer/batcher. Fixtures are built inline against tmp_path so the suite carries
no data files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from senpai.integrations.burr import (
    BurrNight,
    ExecutedCommand,
    RunState,
    parse_burr_filename,
)
from senpai.integrations.burr.night import (
    FrameRecord,
    _attribute_command,
    _cluster_by_time_gap,
    _pointing_key,
    _tracking_mode_from_target,
)

# --- filename parsing ---------------------------------------------------------


class TestParseBurrFilename:
    """Filename parsing across the semantic, UUID, and unrecognized shapes."""

    def test_semantic_calsats(self) -> None:
        """A calsats filename parses task, target, index, timestamp, and verb."""
        p = parse_burr_filename("20260527T071650_calsats_41175_f0.fits")
        assert not p.is_uuid
        assert p.task == "calsats"
        assert p.target == "41175"
        assert p.frame_index == 0
        assert p.timestamp == datetime(2026, 5, 27, 7, 16, 50, tzinfo=UTC)
        assert p.command_verb == "calsat_observed"

    def test_semantic_coverage_alt_az(self) -> None:
        """A coverage AltAz filename parses task, target, index, and verb."""
        p = parse_burr_filename("20260527T071737_coverage_AltAzTarget_f0.fits")
        assert p.task == "coverage"
        assert p.target == "AltAzTarget"
        assert p.frame_index == 0
        assert p.command_verb == "coverage_point_observed"

    def test_semantic_photometric_standards(self) -> None:
        """A multi-word task is preserved verbatim in the parsed record."""
        p = parse_burr_filename(
            "20260527T071926_photometric_standards_ICRSTarget_f0.fits"
        )
        # Multi-word task ('photometric_standards') is preserved verbatim.
        assert p.task == "photometric_standards"
        assert p.target == "ICRSTarget"
        assert p.command_verb == "photometric_standards_observed"

    def test_semantic_rate_subframe(self) -> None:
        """A rate sub-frame parses its target and non-zero frame index."""
        p = parse_burr_filename("20260527T071754_coverage_RateTarget_f1.fits")
        assert p.target == "RateTarget"
        assert p.frame_index == 1

    def test_semantic_calsats_with_underscored_target(self) -> None:
        """A calsats target with an embedded underscore survives parsing."""
        # DAO-01 calsats targets carry a SAT_ prefix + NORAD id; the target must
        # survive the embedded underscore (the old [^_]+ target dropped these).
        p = parse_burr_filename("20260530T022055_calsats_SAT_26605_f0.fits")
        assert p.task == "calsats"
        assert p.target == "SAT_26605"
        assert p.frame_index == 0
        assert p.timestamp == datetime(2026, 5, 30, 2, 20, 55, tzinfo=UTC)

    def test_semantic_photometric_multi_underscore_target(self) -> None:
        """A photometric target with multiple underscores parses whole."""
        p = parse_burr_filename("20260530T022546_photometric_standards_104_485_f1.fits")
        assert p.task == "photometric_standards"
        assert p.target == "104_485"
        assert p.frame_index == 1

    def test_semantic_photometric_messy_target(self) -> None:
        """A photometric target mixing underscores and punctuation parses whole."""
        # Targets like BD_+5_2468 mix underscores and punctuation.
        p = parse_burr_filename("20260530T040313_photometric_standards_BD_+5_2468_f0.fits")
        assert p.task == "photometric_standards"
        assert p.target == "BD_+5_2468"
        assert p.frame_index == 0

    def test_semantic_coverage_numeric_pixel(self) -> None:
        """A coverage frame with a numeric pixel target parses that target."""
        p = parse_burr_filename("20260531T035148_coverage_7_f1.fits")
        assert p.task == "coverage"
        assert p.target == "7"
        assert p.frame_index == 1

    def test_unknown_task_is_unrecognized(self) -> None:
        """An unknown task yields a record with null task and timestamp."""
        # Anchoring on known tasks means an unknown verb falls through rather
        # than being mis-split — caller can fall back to header inspection.
        p = parse_burr_filename("20260530T010000_madeup_task_X_f0.fits")
        assert not p.is_uuid
        assert p.task is None
        assert p.timestamp is None

    def test_uuid_filename(self) -> None:
        """A UUID filename is flagged is_uuid with all semantic fields null."""
        p = parse_burr_filename("0073b353-3f9f-11f1-9659-010101010000.fits")
        assert p.is_uuid
        assert p.timestamp is None
        assert p.task is None
        assert p.frame_index is None
        assert p.command_verb is None

    def test_unrecognized_filename(self) -> None:
        """An unparseable filename returns an all-null, non-UUID record."""
        # Should not raise, returns a record where everything is None and
        # is_uuid is False — caller decides what to do.
        p = parse_burr_filename("garbage_unparseable.fits")
        assert not p.is_uuid
        assert p.timestamp is None
        assert p.task is None
        assert p.command_verb is None

    def test_accepts_path_input(self) -> None:
        """parse_burr_filename accepts a Path and preserves it on the record."""
        p = parse_burr_filename(Path("/some/dir/20260527T071650_calsats_41175_f0.fits"))
        assert p.task == "calsats"
        assert p.path == Path("/some/dir/20260527T071650_calsats_41175_f0.fits")


# --- tracking mode mapping ----------------------------------------------------


class TestTrackingModeFromTarget:
    """Mapping of target tokens to their intended tracking mode."""

    @pytest.mark.parametrize("target,expected", [
        ("AltAzTarget", "sidereal"),
        ("ICRSTarget", "sidereal"),
        ("RateTarget", "rate"),
        ("41175", None),   # NORAD id — adapter must fall back to command log
        (None, None),
    ])
    def test_mapping(self, target: str | None, expected: str | None) -> None:
        """Each target token maps to its expected tracking mode (or None)."""
        assert _tracking_mode_from_target(target) == expected


# --- RunState model -----------------------------------------------------------


class TestRunState:
    """Loading and typed accessors of the RunState model."""

    def test_loads_minimal_real_shape(self, tmp_path: Path) -> None:
        """RunState.load parses a real-shaped run_state with typed accessors."""
        rs_data = {
            "version": "0.3.0",
            "run_id": "Hornet_20260527",
            "observation_date": "2026-05-27T00:00:00-10:00",
            "created_at": "2026-05-27T05:16:43.925464+00:00",
            "config": {
                "site": {
                    "name": "Hornet",
                    "latitude": 10.0,
                    "longitude": 20.0,
                    "altitude_km": 0.1,
                },
                "schedule": {"calsats": {"collect": True}},
            },
            "status": "initialized",
            "current_stage": "setup",
            "executed_commands": [
                {
                    "timestamp": "2026-05-27T07:07:00.498362+00:00",
                    "command": "calsat_observed",
                    "result": "ok",
                    "error": None,
                    "stage": "setup",
                    "metadata": {
                        "observation_time": "2026-05-27T07:06:58.221203+00:00",
                        "satellite_name": "NAVSTAR 62 (USA 201)",
                        "norad_id": 32711,
                        "exposure_time": 2.12,
                        "tracking_modes": ["rate", "rate", "rate", "sidereal"],
                    },
                },
                {
                    "timestamp": "2026-05-27T05:16:44.661532",
                    "command": "run started",
                    "result": "New processing run created",
                    "error": None,
                    "stage": "setup",
                },
            ],
        }
        path = tmp_path / "run_state.json"
        path.write_text(json.dumps(rs_data))
        rs = RunState.load(path)
        assert rs.run_id == "Hornet_20260527"
        assert rs.config.site.latitude == pytest.approx(10.0)
        assert len(rs.executed_commands) == 2
        # collection_commands filters out non-collection events
        coll = rs.collection_commands()
        assert len(coll) == 1
        assert coll[0].command == "calsat_observed"
        # typed accessors on the calsat command
        c = coll[0]
        assert c.tracking_modes == ["rate", "rate", "rate", "sidereal"]
        assert c.target_label == "norad_32711"
        assert c.observation_time == datetime(
            2026, 5, 27, 7, 6, 58, 221203, tzinfo=UTC
        )

    def test_ignores_unknown_top_level_fields(self, tmp_path: Path) -> None:
        """Forward compat: burr controller adds fields, RunState shouldn't break."""
        rs_data = {
            "run_id": "X",
            "newfield_we_dont_know_about": {"a": 1},
            "config": {},
        }
        path = tmp_path / "rs.json"
        path.write_text(json.dumps(rs_data))
        rs = RunState.load(path)
        assert rs.run_id == "X"


# --- attribution --------------------------------------------------------------


def _cmd(verb: str, obs_iso: str, **md: object) -> ExecutedCommand:
    """Build an ExecutedCommand with a given verb and observation time.

    Args:
        verb: The command verb.
        obs_iso: ISO timestamp used for both the command and observation time.
        **md: Extra metadata fields merged into the command metadata.

    Returns:
        An ExecutedCommand at the setup stage.
    """
    return ExecutedCommand(
        timestamp=obs_iso, command=verb, result=None, error=None,
        stage="setup", metadata={"observation_time": obs_iso, **md},
    )


class TestAttributeCommand:
    """Attribution of frames to the correct preceding command."""

    def test_picks_latest_preceding_within_window(self) -> None:
        """A frame attributes to the latest command preceding it in-window."""
        cmds = [
            _cmd("coverage_point_observed", "2026-05-27T07:17:20+00:00", pixel_id=7),
            _cmd("coverage_point_observed", "2026-05-27T07:21:08+00:00", pixel_id=0),
        ]
        # A frame timestamp inside the first command's window picks the first.
        parsed = parse_burr_filename(
            "20260527T071737_coverage_AltAzTarget_f0.fits"
        )
        match = _attribute_command(parsed, cmds, window_s=300.0)
        assert match is cmds[0]

    def test_skips_when_after_window(self) -> None:
        """A frame beyond the attribution window matches no command."""
        cmds = [_cmd("calsat_observed", "2026-05-27T07:00:00+00:00", norad_id=1)]
        parsed = parse_burr_filename("20260527T080000_calsats_1_f0.fits")  # 1h later
        assert _attribute_command(parsed, cmds, window_s=300.0) is None

    def test_skips_when_command_after_frame(self) -> None:
        """A command later than the frame is never attributed."""
        # Frame timestamp predates the command — must never attribute.
        cmds = [_cmd("calsat_observed", "2026-05-27T07:30:00+00:00", norad_id=1)]
        parsed = parse_burr_filename("20260527T070000_calsats_1_f0.fits")
        assert _attribute_command(parsed, cmds, window_s=999_999.0) is None

    def test_does_not_cross_command_types(self) -> None:
        """A frame never binds to a command of a mismatched type."""
        cmds = [_cmd("calsat_observed", "2026-05-27T07:16:00+00:00", norad_id=1)]
        # A coverage frame must never bind to a calsat_observed command.
        parsed = parse_burr_filename(
            "20260527T071737_coverage_AltAzTarget_f0.fits"
        )
        assert _attribute_command(parsed, cmds, window_s=300.0) is None

    def test_picks_closer_when_multiple_match(self) -> None:
        """When several commands match, the closest preceding one wins."""
        cmds = [
            _cmd("coverage_point_observed", "2026-05-27T07:00:00+00:00"),
            _cmd("coverage_point_observed", "2026-05-27T07:17:00+00:00"),
        ]
        parsed = parse_burr_filename(
            "20260527T071800_coverage_AltAzTarget_f0.fits"
        )
        match = _attribute_command(parsed, cmds, window_s=3600.0)
        assert match is cmds[1]


# --- time-gap clustering ------------------------------------------------------


def _record(ts_seconds: int) -> FrameRecord:
    """Build a minimal FrameRecord with just a timestamp offset for clustering.

    The filename's embedded ts is irrelevant — we inject the timestamp directly
    so we can drive gap-based clustering with arbitrary offsets.
    """
    from dataclasses import replace
    base = datetime(2026, 5, 27, 7, 0, 0, tzinfo=UTC)
    parsed = parse_burr_filename(
        "20260527T070000_photometric_standards_ICRSTarget_f0.fits"
    )
    parsed = replace(parsed, timestamp=base + timedelta(seconds=ts_seconds))
    return FrameRecord(path=Path(f"x_{ts_seconds}.fits"), parsed=parsed)


def _offsets(cluster: list[FrameRecord], base: datetime = datetime(2026, 5, 27, 7, 0, 0, tzinfo=UTC)) -> list[int]:
    """Return each record's timestamp offset (in seconds) from a base time.

    Args:
        cluster: The frame records to measure.
        base: Reference time the offsets are computed against.

    Returns:
        The integer second offsets of each record from ``base``.
    """
    return [int((r.parsed.timestamp - base).total_seconds()) for r in cluster]


class TestClusterByTimeGap:
    """Gap-based clustering of frame records by timestamp."""

    def test_empty(self) -> None:
        """An empty input yields no clusters."""
        assert list(_cluster_by_time_gap([], 60.0)) == []

    def test_single(self) -> None:
        """A single record forms one cluster of one."""
        clusters = list(_cluster_by_time_gap([_record(0)], 60.0))
        assert len(clusters) == 1 and len(clusters[0]) == 1

    def test_within_gap_clusters_together(self) -> None:
        """Records within the gap threshold form a single cluster."""
        rs = [_record(0), _record(10), _record(40)]
        clusters = list(_cluster_by_time_gap(rs, 60.0))
        assert len(clusters) == 1
        assert _offsets(clusters[0]) == [0, 10, 40]

    def test_gap_breaks_cluster(self) -> None:
        """A gap larger than the threshold starts a new cluster."""
        rs = [_record(0), _record(10), _record(100)]  # 90s gap between 10 and 100
        clusters = list(_cluster_by_time_gap(rs, 60.0))
        assert len(clusters) == 2
        assert _offsets(clusters[0]) == [0, 10]
        assert _offsets(clusters[1]) == [100]


# --- pointing-key (command-less) batching -------------------------------------


def _rec(task: str, target: str, ts_seconds: int) -> FrameRecord:
    """Build a FrameRecord with a chosen task/target and timestamp offset.

    Exercises the command-less (task, target) clustering path.

    Args:
        task: The task token embedded in the synthetic filename.
        target: The target token embedded in the synthetic filename.
        ts_seconds: Second offset applied to the record's timestamp.

    Returns:
        A FrameRecord carrying the chosen task/target and adjusted timestamp.
    """
    from dataclasses import replace
    base = datetime(2026, 5, 30, 2, 0, 0, tzinfo=UTC)
    parsed = parse_burr_filename(f"20260530T020000_{task}_{target}_f0.fits")
    parsed = replace(parsed, timestamp=base + timedelta(seconds=ts_seconds))
    return FrameRecord(path=Path(f"{task}_{target}_{ts_seconds}.fits"), parsed=parsed)


class TestPointingKey:
    """Derivation of a pointing-identity key from a frame record."""

    def test_mode_token_target_has_no_pointing_identity(self) -> None:
        """Mode-token targets carry no pointing identity and key to None."""
        # AltAzTarget/RateTarget encode tracking mode, not pointing → None.
        assert _pointing_key(_rec("coverage", "AltAzTarget", 0)) is None
        assert _pointing_key(_rec("photometric_standards", "RateTarget", 0)) is None

    def test_concrete_target_is_pointing_identity(self) -> None:
        """A concrete target token is used directly as the pointing key."""
        assert _pointing_key(_rec("calsats", "SAT_26605", 0)) == "SAT_26605"
        assert _pointing_key(_rec("coverage", "7", 0)) == "7"


class TestClusterByPointing:
    """Pointing-aware clustering via the pointing-key on gap clustering."""

    def test_target_change_breaks_within_gap(self) -> None:
        """A target change breaks a cluster even within the time gap."""
        # Two calsat sequences 10s apart — well within the 60s gap — but
        # different NORAD ids must land in separate batches.
        rs = [
            _rec("calsats", "SAT_1", 0),
            _rec("calsats", "SAT_1", 7),
            _rec("calsats", "SAT_2", 17),
            _rec("calsats", "SAT_2", 24),
        ]
        clusters = list(_cluster_by_time_gap(rs, 60.0, key=_pointing_key))
        assert [len(c) for c in clusters] == [2, 2]
        assert {r.parsed.target for r in clusters[0]} == {"SAT_1"}
        assert {r.parsed.target for r in clusters[1]} == {"SAT_2"}

    def test_same_target_far_apart_still_splits_on_gap(self) -> None:
        """The same target revisited after a long gap still splits into two."""
        # Same coverage pixel revisited after a long slew → two batches.
        rs = [_rec("coverage", "7", 0), _rec("coverage", "7", 5000)]
        clusters = list(_cluster_by_time_gap(rs, 60.0, key=_pointing_key))
        assert [len(c) for c in clusters] == [1, 1]

    def test_none_key_falls_back_to_pure_gap(self) -> None:
        """A None pointing key does not force a break; pure-gap behavior holds."""
        # Mode-token targets alternate within one pointing; a None key must not
        # force a break (preserves the photometric_standards orphan behavior).
        rs = [
            _rec("photometric_standards", "ICRSTarget", 0),
            _rec("photometric_standards", "RateTarget", 8),
            _rec("photometric_standards", "RateTarget", 15),
        ]
        clusters = list(_cluster_by_time_gap(rs, 60.0, key=_pointing_key))
        assert len(clusters) == 1 and len(clusters[0]) == 3


# --- BurrNight end-to-end -----------------------------------------------------


def _make_night(
    tmp_path: Path,
    *,
    files: list[str],
    commands: list[dict],
) -> tuple[BurrNight, Path]:
    """Spin up a tmp_path layout that mirrors /burr/{Hornet/, burr/Hornet_XXXX/}."""
    burr_root = tmp_path / "burr_root"
    sensor_dir = burr_root / "Hornet"
    night_meta = burr_root / "burr" / "Hornet_20260527" / "metadata"
    sensor_dir.mkdir(parents=True)
    night_meta.mkdir(parents=True)

    for name in files:
        (sensor_dir / name).touch()

    rs = {
        "version": "0.3.0",
        "run_id": "Hornet_20260527",
        "observation_date": "2026-05-27T00:00:00-10:00",
        "config": {"site": {"name": "H", "latitude": 0, "longitude": 0}},
        "lighting_schedule": {
            "night_start": "2026-05-27T05:00:00+00:00",
            "night_end": "2026-05-27T14:00:00+00:00",
        },
        "executed_commands": commands,
    }
    (night_meta / "run_state.json").write_text(json.dumps(rs))

    night = BurrNight.from_night_dir(burr_root / "burr" / "Hornet_20260527")
    return night, sensor_dir


def test_burrnight_calsat_batch_grouping(tmp_path: Path) -> None:
    """A calsat sequence groups into one batch with per-frame tracking modes."""
    files = [
        "20260527T071650_calsats_41175_f0.fits",
        "20260527T071658_calsats_41175_f1.fits",
        "20260527T071706_calsats_41175_f2.fits",
        "20260527T071720_calsats_41175_f3.fits",
    ]
    cmd = {
        "timestamp": "2026-05-27T07:16:35+00:00",
        "command": "calsat_observed",
        "stage": "setup",
        "result": "ok",
        "metadata": {
            "observation_time": "2026-05-27T07:16:35+00:00",
            "satellite_name": "GALILEO 11 (268)",
            "norad_id": 41175,
            "exposure_time": 3.89,
            "tracking_modes": ["rate", "rate", "rate", "sidereal"],
        },
    }
    night, _ = _make_night(tmp_path, files=files, commands=[cmd])
    batches = list(night.frame_batches())
    assert len(batches) == 1
    b = batches[0]
    assert b.command is cmd or b.command.command == "calsat_observed"
    assert len(b.frames) == 4
    # f0..f3 → modes from tracking_modes vector for calsats
    assert [f.intended_tracking_mode for f in b.frames] == [
        "rate", "rate", "rate", "sidereal",
    ]


def test_burrnight_coverage_uses_target_token_for_mode(tmp_path: Path) -> None:
    """Coverage frames derive tracking mode from the target token, not the vector."""
    files = [
        "20260527T071737_coverage_AltAzTarget_f0.fits",   # sidereal sub-exposure
        "20260527T071747_coverage_RateTarget_f0.fits",    # rate, sub-frame 0
        "20260527T071754_coverage_RateTarget_f1.fits",    # rate, sub-frame 1
    ]
    cmd = {
        "timestamp": "2026-05-27T07:17:20+00:00",
        "command": "coverage_point_observed",
        "stage": "setup",
        "result": "ok",
        "metadata": {
            "observation_time": "2026-05-27T07:17:20+00:00",
            "map_id": 1, "pixel_id": 7,
            "exposure_times": [1.0, 3.0, 5.0],
            "tracking_modes": ["sidereal", "rate", "rate"],
        },
    }
    night, _ = _make_night(tmp_path, files=files, commands=[cmd])
    batches = list(night.frame_batches())
    assert len(batches) == 1
    modes = [f.intended_tracking_mode for f in batches[0].frames]
    # Target tokens drive mode; not the tracking_modes vector + frame_index.
    assert sorted(modes) == ["rate", "rate", "sidereal"]


def test_burrnight_clusters_orphans_per_pointing(tmp_path: Path) -> None:
    """Command-less frames cluster into per-pointing batches by time gap."""
    # No command logged (photometric_standards), but timestamps cluster into
    # two pointings separated by >60s.
    files = [
        "20260527T071926_photometric_standards_ICRSTarget_f0.fits",
        "20260527T071935_photometric_standards_RateTarget_f0.fits",
        "20260527T071942_photometric_standards_RateTarget_f1.fits",
        # 251s gap → new pointing
        "20260527T072413_photometric_standards_ICRSTarget_f0.fits",
        "20260527T072425_photometric_standards_RateTarget_f0.fits",
    ]
    night, _ = _make_night(tmp_path, files=files, commands=[])
    batches = list(night.frame_batches())
    assert len(batches) == 2
    sizes = sorted(len(b.frames) for b in batches)
    assert sizes == [2, 3]


def test_burrnight_skips_out_of_window_frames(tmp_path: Path) -> None:
    """Semantic frames outside the night window are dropped from the index."""
    files = [
        "20250101T010000_calsats_1_f0.fits",  # way before night window
        "20260527T071650_calsats_41175_f0.fits",  # in-window
    ]
    night, _ = _make_night(tmp_path, files=files, commands=[])
    records = night.index_frames()
    # Out-of-window semantic frames are dropped; uuid would be kept but there
    # are none here.
    paths = [r.path.name for r in records]
    assert "20260527T071650_calsats_41175_f0.fits" in paths
    assert "20250101T010000_calsats_1_f0.fits" not in paths


def test_auto_nights_splits_flat_multi_night_dir(tmp_path: Path) -> None:
    """A flat two-night dir splits into two BurrNights.

    Splits into evening-local ids, frame-derived windows, and command-less
    (task, target) batching (covers burr's "didn't split per night" bug).
    """
    data_dir = tmp_path / "DAO-01"
    meta_dir = tmp_path / "processed" / "DAO-01_20260528" / "metadata"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    # Night A: UTC 2026-05-30 ~02:00 (evening-local 05-29 at UTC-4).
    # Night B: UTC 2026-05-31 ~02:00 (evening-local 05-30). 24h gap between.
    files = [
        # night A: one calsat sequence + a second calsat (different NORAD)
        "20260530T020000_calsats_SAT_100_f0.fits",
        "20260530T020007_calsats_SAT_100_f1.fits",
        "20260530T020030_calsats_SAT_200_f0.fits",
        "20260530T020037_calsats_SAT_200_f1.fits",
        # night B: one coverage pixel
        "20260531T020000_coverage_5_f0.fits",
        "20260531T020010_coverage_5_f1.fits",
    ]
    for name in files:
        (data_dir / name).touch()

    rs = {
        "version": "0.3.0",
        "run_id": "DAO-01_20260528",
        # tz offset here drives evening-local night labeling
        "observation_date": "2026-05-28T00:00:00-04:00",
        "config": {"site": {"name": "DAO-01", "latitude": 10.0, "longitude": 20.0}},
        # lighting_schedule describes a *different* night and must be ignored
        "lighting_schedule": {
            "night_start": "2026-05-29T03:06:00+00:00",
            "night_end": "2026-05-29T07:59:00+00:00",
        },
        "executed_commands": [],
    }
    (meta_dir / "run_state.json").write_text(json.dumps(rs))

    nights = BurrNight.auto_nights(meta_dir / "run_state.json", data_dir)
    assert [n.night_id for n in nights] == ["DAO-01_20260529", "DAO-01_20260530"]

    night_a, night_b = nights
    # Window is frame-derived, not the (mismatched) lighting_schedule.
    assert night_a.window_start < datetime(2026, 5, 30, 2, 0, tzinfo=UTC)
    assert night_a.window_end > datetime(2026, 5, 30, 2, 0, tzinfo=UTC)

    batches_a = list(night_a.frame_batches())
    # Two calsats split by NORAD despite being 23s apart (no command log).
    assert len(batches_a) == 2
    assert all(b.command is None for b in batches_a)
    assert all(len(b.frames) == 2 for b in batches_a)

    batches_b = list(night_b.frame_batches())
    assert len(batches_b) == 1
    assert len(batches_b[0].frames) == 2


def test_frame_batches_by_seq_key(tmp_path: Path) -> None:
    """Batching by a FITS-header seq_key groups one batch per set.

    Frames missing the keyword fall back to command/orphan batching rather than
    being dropped.
    """
    import numpy as np
    from astropy.io import fits

    data_dir = tmp_path / "DAO-01"
    meta_dir = tmp_path / "processed" / "DAO-01_20260528" / "metadata"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    def w(name: str, seq: str | None) -> None:
        """Write a 2x2 FITS frame, optionally stamping a BURRSEQ header.

        Args:
            name: Frame file name to write into the data directory.
            seq: BURRSEQ value to stamp, or None to omit the keyword.
        """
        hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.int16))
        if seq is not None:
            hdu.header["BURRSEQ"] = seq
        hdu.writeto(data_dir / name)

    # Two BURRSEQ sets at the same coverage pointing, plus one frame that is
    # missing the keyword entirely.
    w("20260530T020000_coverage_3_f0.fits", "SEQ-A")
    w("20260530T020005_coverage_3_f1.fits", "SEQ-A")
    w("20260530T020010_coverage_3_f0.fits", "SEQ-B")
    w("20260530T020015_coverage_3_f1.fits", "SEQ-B")
    w("20260530T020020_coverage_3_f0.fits", None)

    rs = {
        "run_id": "DAO-01_20260528",
        "observation_date": "2026-05-28T00:00:00-04:00",
        "config": {"site": {"name": "DAO-01", "latitude": 10.0, "longitude": 20.0}},
        "executed_commands": [],
    }
    (meta_dir / "run_state.json").write_text(json.dumps(rs))

    night = BurrNight.auto_nights(meta_dir / "run_state.json", data_dir)[0]
    batches = list(night.frame_batches(seq_key="BURRSEQ"))

    seq_batches = [b for b in batches if all(f.seq_id for f in b.frames)]
    assert len(seq_batches) == 2
    assert sorted(len(b.frames) for b in seq_batches) == [2, 2]
    assert {f.seq_id for b in seq_batches for f in b.frames} == {"SEQ-A", "SEQ-B"}

    # The keyword-less frame is not dropped — it surfaces via the fallback path.
    assert sum(len(b.frames) for b in batches) == 5


def test_burrnight_keeps_uuid_records_without_attribution(tmp_path: Path) -> None:
    """UUID-named frames are indexed but left unattributed to any command."""
    files = [
        "0073b353-3f9f-11f1-9659-010101010000.fits",
        "20260527T071650_calsats_41175_f0.fits",
    ]
    night, _ = _make_night(tmp_path, files=files, commands=[])
    records = night.index_frames()
    uuid_records = [r for r in records if r.parsed.is_uuid]
    assert len(uuid_records) == 1
    assert uuid_records[0].command is None
