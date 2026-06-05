"""Tests for COCO export (senpai.export.coco) and dataset splitting
(senpai.export.dataset_split).

Synthetic ``SenpaiRunResult`` objects are built with serializable sidereal and
rate frames that point at small FITS files written into ``tmp_path``. Stars and
satellites are placed at known pixel positions so the generated COCO bboxes and
streak lines can be checked exactly. All output goes to ``tmp_path``; nothing
touches the network or a GUI, and the run is deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from senpai.export.coco import SenpaiCocoExporter
from senpai.export.dataset_split import (
    DatasetSplit,
    DatasetSplitter,
    split_coco_dataset,
)

# ---------------------------------------------------------------------------
# Builders for synthetic SENPAI run data
# ---------------------------------------------------------------------------


def _write_fits(path, width=64, height=48):
    """Write a tiny FITS file and return (path, width, height)."""
    data = np.zeros((height, width), dtype=np.float32)
    header = fits.Header()
    header["EXPTIME"] = 2.5
    header["GAIN"] = 1.3
    header["DATE-OBS"] = "2024-01-02T03:04:05"
    fits.PrimaryHDU(data, header).writeto(path, overwrite=True)
    return str(path), width, height


def _image_metadata(width, height):
    from senpai.engine.models.metadata import ImageMetadata

    return ImageMetadata(image_id="img", width=width, height=height)


def _starfield(width, height, *, fit=True, detections=None, catalog=None):
    from senpai.engine.models.starfield import StarField, StarInImage, StarInSpace

    return StarField(
        detections=[StarInImage(**d) for d in (detections or [])],
        catalog_stars=([StarInSpace(**c) for c in catalog] if catalog is not None else None),
        image_metadata=_image_metadata(width, height),
        fit=fit,
        wcs=None,
    )


def _sidereal_frame(fits_path, width, height, *, index=0, detections=None, catalog=None, fit=True):
    from senpai.engine.models.senpai import SiderealFrameSerializable

    return SiderealFrameSerializable(
        starfield=_starfield(width, height, fit=fit, detections=detections, catalog=catalog),
        processed_frame_path=fits_path,
        index=index,
        timestamp="2024-01-02T03:04:05",
    )


def _streak(length=20.0, angle_deg=0.0, fwhm=3.0):
    from senpai.engine.models.metadata import StreakMetadata

    rad = np.deg2rad(angle_deg)
    return StreakMetadata(
        pixel_length=length,
        sine_angle=float(np.sin(rad)),
        cosine_angle=float(np.cos(rad)),
        fwhm=fwhm,
    )


def _satellite(x, y, snr=12.0):
    from senpai.engine.models.starfield import SatelliteInImage

    return SatelliteInImage(x=x, y=y, snr=snr)


def _rate_frame(fits_path, width, height, *, index=0, satellites=None, detections=None, streak=None, fit=True):
    from senpai.engine.models.senpai import RateTrackFrameSerializable
    from senpai.engine.models.starfield import SatelliteListImage

    sat_list = None
    if satellites is not None:
        sat_list = SatelliteListImage(
            detections=satellites,
            image_metadata=_image_metadata(width, height),
        )

    return RateTrackFrameSerializable(
        starfield=_starfield(width, height, fit=fit, detections=detections),
        streak=streak,
        detections=sat_list,
        processed_frame_path=fits_path,
        index=index,
        timestamp="2024-01-02T03:04:05",
    )


def _run(sidereal=None, rate=None, scale_factor=None):
    from senpai.engine.models.metadata import CollectionMetadata
    from senpai.engine.models.senpai import SenpaiRunResult

    return SenpaiRunResult(
        id="run-1",
        num_frames=len(sidereal or []) + len(rate or []),
        collect_metadata=CollectionMetadata(),
        completed=True,
        scale_factor=scale_factor,
        sidereal_frames=sidereal or [],
        rate_track_frames=rate or [],
    )


# ---------------------------------------------------------------------------
# Sidereal point-source export
# ---------------------------------------------------------------------------


def test_sidereal_point_annotations_bbox(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "side.fits")
    # Two well-separated detections at known positions.
    dets = [
        {"x": 10.0, "y": 20.0, "counts": 50_000.0},
        {"x": 40.0, "y": 30.0, "counts": 80_000.0},
    ]
    run = _run(sidereal=[_sidereal_frame(fits_path, w, h, detections=dets)])

    out = tmp_path / "coco"
    exporter = SenpaiCocoExporter(output_dir=out, write_fits=False, box_size=6)
    exporter.export_senpai_run(run, collect_id="C1")

    point_file = out / "C1_sidereal_0_point_sat.json"
    assert point_file.exists()
    data = json.loads(point_file.read_text())

    # Categories
    assert data["categories"] == [{"id": 0, "name": "sidereal_star", "supercategory": "point_source"}]
    # One image entry, dimensions from the FITS array.
    assert len(data["images"]) == 1
    img = data["images"][0]
    assert img["width"] == w
    assert img["height"] == h
    assert img["type"] == "sidereal"
    assert img["id"] == "C1_sidereal_0"

    # Two annotations, bbox centered on each detection (box_size=6 -> offset 3).
    anns = data["annotations"]
    assert len(anns) == 2
    by_centroid = {tuple(a["centroid"]): a for a in anns}
    assert (10.0, 20.0) in by_centroid
    bbox = by_centroid[(10.0, 20.0)]["bbox"]
    assert bbox == [10.0 - 3, 20.0 - 3, 6, 6]
    assert by_centroid[(10.0, 20.0)]["area"] == 36


def test_sidereal_skipped_when_no_wcs(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "side.fits")
    dets = [{"x": 5.0, "y": 5.0, "counts": 10_000.0}]
    run = _run(sidereal=[_sidereal_frame(fits_path, w, h, detections=dets, fit=False)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False).export_senpai_run(run, collect_id="C1")

    # No COCO files produced for an unsolved frame.
    assert not list(out.glob("*.json"))


def test_process_sidereal_flag_disables_sidereal(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "side.fits")
    dets = [{"x": 5.0, "y": 5.0, "counts": 10_000.0}]
    run = _run(sidereal=[_sidereal_frame(fits_path, w, h, detections=dets)])

    out = tmp_path / "coco"
    exporter = SenpaiCocoExporter(output_dir=out, write_fits=False, process_sidereal=False)
    exporter.export_senpai_run(run, collect_id="C1")
    assert not list(out.glob("*.json"))


def test_snr_cut_filters_low_snr_detections(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "side.fits")
    # counts -> snr via sqrt(counts/1000); 1e6 counts -> snr ~31.6 (kept),
    # 10 counts -> snr ~0.1 (cut at snr_cut=5).
    dets = [
        {"x": 12.0, "y": 12.0, "counts": 1_000_000.0},
        {"x": 30.0, "y": 30.0, "counts": 10.0},
    ]
    run = _run(sidereal=[_sidereal_frame(fits_path, w, h, detections=dets)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False, snr_cut=5.0).export_senpai_run(run, collect_id="C1")

    data = json.loads((out / "C1_sidereal_0_point_sat.json").read_text())
    centroids = {tuple(a["centroid"]) for a in data["annotations"]}
    assert (12.0, 12.0) in centroids
    assert (30.0, 30.0) not in centroids


# ---------------------------------------------------------------------------
# Rate-track export: satellites + streak lines
# ---------------------------------------------------------------------------


def test_rate_satellite_and_streak_annotations(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "rate.fits")
    sats = [_satellite(25.0, 15.0, snr=20.0)]
    star_dets = [{"x": 30.0, "y": 24.0, "counts": 500_000.0}]
    streak = _streak(length=20.0, angle_deg=0.0)
    run = _run(rate=[_rate_frame(fits_path, w, h, satellites=sats, detections=star_dets, streak=streak)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False, streak_box_size=8).export_senpai_run(run, collect_id="C2")

    # Satellite (point) annotations.
    sat_file = out / "C2_rate_0_point_sat.json"
    assert sat_file.exists()
    sat_data = json.loads(sat_file.read_text())
    assert sat_data["categories"][0]["name"] == "satellite"
    assert len(sat_data["annotations"]) == 1
    assert sat_data["annotations"][0]["centroid"] == [25.0, 15.0]
    # streak_box_size=8 -> bbox offset 4
    assert sat_data["annotations"][0]["bbox"] == [25.0 - 4, 15.0 - 4, 8, 8]

    # Streak (line) annotations.
    line_file = out / "C2_rate_0_line_star.json"
    assert line_file.exists()
    line_data = json.loads(line_file.read_text())
    assert line_data["categories"][0]["name"] == "rate_star"
    assert line_data["categories"][0]["supercategory"] == "streak_source"
    assert len(line_data["annotations"]) == 1
    ann = line_data["annotations"][0]
    assert ann["type"] == "line"
    # Horizontal streak length 20 centered on star (30, 24):
    # start = (30 - 10, 24), direction = (20, 0).
    line = ann["line"]
    assert line == [20.0, 24.0, 20.0, 0.0]


def test_rate_streak_line_angle(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "rate.fits")
    star_dets = [{"x": 32.0, "y": 24.0, "counts": 500_000.0}]
    streak = _streak(length=10.0, angle_deg=90.0)  # vertical
    run = _run(rate=[_rate_frame(fits_path, w, h, detections=star_dets, streak=streak)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False).export_senpai_run(run, collect_id="C2")

    line_data = json.loads((out / "C2_rate_0_line_star.json").read_text())
    line = line_data["annotations"][0]["line"]
    # 90 deg: cos~0, sin~1. start = (32, 24-5), dir = (0, 10).
    assert line[0] == pytest.approx(32.0, abs=1e-3)
    assert line[1] == pytest.approx(19.0, abs=1e-3)
    assert line[2] == pytest.approx(0.0, abs=1e-3)
    assert line[3] == pytest.approx(10.0, abs=1e-3)


def test_rate_skipped_when_streak_exceeds_max_length(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "rate.fits")
    star_dets = [{"x": 30.0, "y": 24.0, "counts": 500_000.0}]
    streak = _streak(length=200.0)
    run = _run(rate=[_rate_frame(fits_path, w, h, detections=star_dets, streak=streak)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False, max_streak_length=50.0).export_senpai_run(run, collect_id="C2")
    assert not list(out.glob("*.json"))


def test_rate_skipped_when_no_wcs(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "rate.fits")
    sats = [_satellite(10.0, 10.0)]
    run = _run(rate=[_rate_frame(fits_path, w, h, satellites=sats, fit=False)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False).export_senpai_run(run, collect_id="C2")
    assert not list(out.glob("*.json"))


def test_scale_factor_scales_coordinates(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "side.fits", width=64, height=64)
    dets = [{"x": 10.0, "y": 10.0, "counts": 1_000_000.0}]
    run = _run(sidereal=[_sidereal_frame(fits_path, w, h, detections=dets)], scale_factor=2.0)

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False).export_senpai_run(run, collect_id="C3")

    data = json.loads((out / "C3_sidereal_0_point_sat.json").read_text())
    # Detection at (10,10) scaled by 2 -> (20, 20).
    assert data["annotations"][0]["centroid"] == [20.0, 20.0]


def test_write_fits_creates_image_file(tmp_path):
    fits_path, w, h = _write_fits(tmp_path / "side.fits")
    dets = [{"x": 10.0, "y": 10.0, "counts": 1_000_000.0}]
    run = _run(sidereal=[_sidereal_frame(fits_path, w, h, detections=dets)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=True).export_senpai_run(run, collect_id="C4")

    written = out / "C4_sidereal_0.fits"
    assert written.exists()
    data = json.loads((out / "C4_sidereal_0_point_sat.json").read_text())
    assert data["images"][0]["file_name"] == "C4_sidereal_0.fits"


def test_export_batch_processes_multiple_runs(tmp_path):
    fp1, w, h = _write_fits(tmp_path / "a.fits")
    fp2, _, _ = _write_fits(tmp_path / "b.fits")
    dets = [{"x": 10.0, "y": 10.0, "counts": 1_000_000.0}]
    run1 = _run(sidereal=[_sidereal_frame(fp1, w, h, detections=dets)])
    run2 = _run(sidereal=[_sidereal_frame(fp2, w, h, detections=dets)])

    out = tmp_path / "coco"
    SenpaiCocoExporter(output_dir=out, write_fits=False).export_batch([run1, run2], ["A", "B"])
    assert (out / "A_sidereal_0_point_sat.json").exists()
    assert (out / "B_sidereal_0_point_sat.json").exists()


# ---------------------------------------------------------------------------
# dataset_split
# ---------------------------------------------------------------------------


def test_dataset_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        DatasetSplit(train=0.5, val=0.2, test=0.1)


def test_dataset_split_defaults_valid():
    s = DatasetSplit()
    assert (s.train, s.val, s.test) == (0.7, 0.2, 0.1)


def _make_point_annotation_file(input_dir, image_id, file_name, frame_type, n_ann=1):
    data = {
        "images": [
            {
                "file_name": file_name,
                "width": 16,
                "height": 16,
                "id": image_id,
                "type": frame_type,
            }
        ],
        "annotations": [{"id": i, "image_id": image_id, "type": "bbox", "bbox": [0, 0, 4, 4]} for i in range(n_ann)],
        "categories": [{"id": 0, "name": "satellite", "supercategory": "point_source"}],
    }
    (input_dir / f"{image_id}_point_sat.json").write_text(json.dumps(data))


def _touch_image(input_dir, file_name):
    np.zeros((4, 4), dtype=np.float32)
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(input_dir / file_name, overwrite=True)


def test_dataset_split_raises_without_annotations(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    splitter = DatasetSplitter(DatasetSplit(), random_seed=0)
    with pytest.raises(ValueError):
        splitter.split_coco_dataset(input_dir, tmp_path / "out")


def test_dataset_split_deterministic_with_seed(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    # 10 images with parseable timestamps in the filename so temporal sort is stable.
    for i in range(10):
        fname = f"frame_2024010{i % 10}_000000_{i}.fits"
        image_id = f"img_{i}"
        _make_point_annotation_file(input_dir, image_id, fname, "rate")
        _touch_image(input_dir, fname)

    def run(seed):
        out = tmp_path / f"out_{seed}"
        return split_coco_dataset(
            str(input_dir),
            str(out),
            random_seed=seed,
            temporal_split=False,
        )

    r1 = run(123)
    r2 = run(123)
    assert {k: sorted(v) for k, v in r1.items()} == {k: sorted(v) for k, v in r2.items()}
    # All 10 images accounted for across splits with no overlap.
    all_ids = r1["train"] + r1["val"] + r1["test"]
    assert sorted(all_ids) == sorted(f"img_{i}" for i in range(10))
    assert len(set(all_ids)) == 10


def test_dataset_split_sizes_match_ratios(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(10):
        fname = f"frame_{i}.fits"
        image_id = f"img_{i}"
        _make_point_annotation_file(input_dir, image_id, fname, "rate")
        _touch_image(input_dir, fname)

    out = tmp_path / "out"
    result = split_coco_dataset(
        str(input_dir),
        str(out),
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=0,
        temporal_split=False,
    )
    assert len(result["train"]) == 7
    assert len(result["val"]) == 2
    assert len(result["test"]) == 1


def test_dataset_split_writes_combined_annotation_files(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(4):
        fname = f"frame_{i}.fits"
        image_id = f"img_{i}"
        _make_point_annotation_file(input_dir, image_id, fname, "rate")
        _touch_image(input_dir, fname)

    out = tmp_path / "out"
    split_coco_dataset(str(input_dir), str(out), random_seed=1, temporal_split=False)

    ann_dir = out / "annotations"
    for split_name in ("train", "val", "test"):
        assert (ann_dir / f"points_{split_name}.json").exists()
        assert (ann_dir / f"lines_{split_name}.json").exists()
    points_train = json.loads((ann_dir / "points_train.json").read_text())
    assert points_train["categories"][0]["name"] == "satellite"


def test_dataset_split_lines_excludes_sidereal(tmp_path):
    """Lines dataset only includes rate-type images/annotations."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    # One sidereal image with a point annotation only.
    _make_point_annotation_file(input_dir, "sid_0", "sid_0.fits", "sidereal")
    _touch_image(input_dir, "sid_0.fits")
    # One rate image with a line annotation file.
    rate_data = {
        "images": [{"file_name": "rate_0.fits", "width": 16, "height": 16, "id": "rate_0", "type": "rate"}],
        "annotations": [{"id": 0, "image_id": "rate_0", "type": "line", "line": [0, 0, 4, 0]}],
        "categories": [{"id": 0, "name": "rate_star", "supercategory": "streak_source"}],
    }
    (input_dir / "rate_0_line_star.json").write_text(json.dumps(rate_data))
    _touch_image(input_dir, "rate_0.fits")
    # And a point annotation for the rate frame so it appears in point images too.
    _make_point_annotation_file(input_dir, "rate_0", "rate_0.fits", "rate")

    out = tmp_path / "out"
    # train=1.0 so every image lands in the train split deterministically.
    split = DatasetSplit(train=1.0, val=0.0, test=0.0)
    DatasetSplitter(split, random_seed=0).split_coco_dataset(input_dir, out, temporal_split=False)

    lines_train = json.loads((out / "annotations" / "lines_train.json").read_text())
    line_image_ids = {img["id"] for img in lines_train["images"]}
    assert "sid_0" not in line_image_ids
    assert "rate_0" in line_image_ids
    # All line annotations belong to rate frames.
    assert all(a["image_id"] == "rate_0" for a in lines_train["annotations"])
