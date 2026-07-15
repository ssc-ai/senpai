"""Shared CLI utilities for command/config saving, profiling, and serialization."""

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


def save_run_metadata(output_dir: Path, module_name: str, config) -> None:
    """Save command.txt and config.yaml to output_dir for reproducibility."""
    import yaml

    output_dir = Path(output_dir)

    # command.txt: sys.argv with module_name replacing argv[0]
    command_args = sys.argv[:]
    command_args[0] = f"python -m {module_name}"
    command_file = output_dir / "command.txt"
    with open(command_file, "w") as f:
        f.write(" ".join(command_args))
    logger.info("Command saved to: %s", command_file)

    # config.yaml: wrap under "app" key to match the format expected by load_yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump({"app": config.model_dump(mode="json")}, f, default_flow_style=False)
    logger.info("Config saved to: %s", output_dir / "config.yaml")


# Per-star / completeness arrays are dropped from the per-frame quick-look
# files; they stay in the run summary and full result JSONs. The quick-look is
# for eyeballing detections + WCS, not photometric analysis.
_QUICKLOOK_PHOTOMETRY_DROP = frozenset({
    "stars_mag",
    "stars_snr",
    "stars_zp_offset",
    "stars_isolated",
    "stars_catalog_id",
    "completeness_mag",
    "completeness_pct",
})


def write_frame_quicklooks(summary, output_dir: Path) -> None:
    """Write compact per-frame quick-look JSONs (frame_{index}_{mode}.json).

    Takes a SenpaiRunSummary: each frame gets its FrameSummary (detections,
    WCS, streaks, scalar photometry) minus the bulk per-star arrays.
    """
    for fs in summary.frames:
        data = fs.model_dump(mode="json")
        ps = data.get("photometry_summary")
        if ps:
            data["photometry_summary"] = {
                k: v for k, v in ps.items() if k not in _QUICKLOOK_PHOTOMETRY_DROP
            }
        mode = fs.track_mode or "frame"
        with open(Path(output_dir) / f"frame_{fs.index}_{mode}.json", "w") as f:
            json.dump(data, f)
    logger.info("Wrote %d per-frame quick-look JSONs", len(summary.frames))


def profile_run(func, *args, run_id: str = "profile", **kwargs):
    """Generic profiling wrapper. Runs func(*args, **kwargs) under cProfile,
    saves top-30 stats to output_dir/profile_{run_id}.txt, returns func's result."""
    import cProfile
    import io
    import pstats
    from pstats import SortKey

    from senpai.core.config import get_config

    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)

    config = get_config()
    with open(config.runtime.output_dir / f"profile_{run_id}.txt", "w") as f:
        f.write(s.getvalue())

    logger.info("Profile results saved to profile_%s.txt", run_id)

    return result


def serialize_photometry_to_json(results, summary, output_path: Path) -> None:
    """Serialize photometry results + summary to JSON.

    Uses dataclasses.asdict for summary, result.star.model_dump() for Pydantic models.
    """
    photometry_output = {
        "summary": asdict(summary),
        "results": [
            {
                **asdict(result),
                "star": result.star.model_dump(),  # StarInSpace is a Pydantic model
            }
            for result in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(photometry_output, f)
    logger.info("Photometry results saved to %s", output_path)


def ensure_output_dir(output_dir: Path, default_stem: str | None = None) -> Path:
    """Standardized output directory creation. If output_dir is '.', uses default_stem."""
    output_dir = Path(output_dir)
    if output_dir == Path(".") and default_stem is not None:
        output_dir = Path(default_stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
