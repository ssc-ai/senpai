# SENPAI Export Module

This module provides functionality to export SENPAI run data to COCO format for machine learning applications.

## Overview

The export module converts SENPAI astronomical data processing results into individual COCO format files per image, making it suitable for training computer vision models for satellite detection and astronomical object recognition.

## Features

- **Individual COCO files**: Creates separate COCO JSON files for each image with point source and streak annotations
- **Multiple image formats**: Supports both FITS and PNG image export
- **Annotated images**: Option to save images with overlaid annotations
- **Flexible filtering**: Configurable SNR cuts, mask radius, and other parameters
- **Batch processing**: Export multiple SENPAI runs from folder structures
- **Dataset splitting**: Split exported datasets into train/val/test sets

## Usage

### Command Line Interface

#### Export a single SENPAI run:
```bash
python -m senpai.export.cli single /path/to/run.json /path/to/output \
  --write-fits --save-annotated-images --snr-cut 2.0 --verbose
```

#### Export all runs from a folder:
```bash
python -m senpai.export.cli folder /path/to/senpai/data /path/to/output \
  --max-runs 10 --write-fits --save-annotated-images --verbose
```

### Example Script

Use the provided example script for quick folder exports:

```bash
python senpai/export/example_folder.py /path/to/senpai/data /path/to/output \
  --max-runs 5 --write-fits --save-annotated-images --verbose
```

### Programmatic Usage

```python
from senpai.export.coco import SenpaiCocoExporter
from senpai.engine.utils.file_io import load_senpai_run

# Load a SENPAI run
senpai_run = load_senpai_run("/path/to/run.json")

# Create exporter
exporter = SenpaiCocoExporter(
    output_dir="/path/to/output",
    write_fits=True,
    save_annotated_images=True,
    snr_cut=2.0,
    box_size=4,
    streak_box_size=10,
)

# Export the run
exporter.export_senpai_run(senpai_run, "observation_001")
```

## Output Format

The exporter creates individual COCO format files for each image:

### File Structure
```
output_dir/
├── observation_001_sidereal_0.fits          # FITS image
├── observation_001_sidereal_0_point_sat.json # Point source annotations
├── observation_001_rate_1.fits              # Rate track image
├── observation_001_rate_1_point_sat.json    # Satellite annotations
├── observation_001_rate_1_line_star.json    # Streak annotations
├── observation_001_rate_1_annotated.png     # Annotated image (if enabled)
└── ...
```

### COCO Format

Each JSON file follows the COCO format with:

- **Images**: Image metadata including dimensions, exposure time, gain, etc.
- **Annotations**: Detection annotations with bounding boxes or line segments
- **Categories**: Object categories (point sources, streaks, satellites)

#### Point Source Annotations
```json
{
  "id": 0,
  "centroid": [512.5, 384.2],
  "bbox": [510.5, 382.2, 4, 4],
  "image_id": "observation_001_sidereal_0",
  "type": "bbox",
  "snr": 15.2,
  "vmag": 12.5,
  "area": 16
}
```

#### Streak Annotations
```json
{
  "image_id": "observation_001_rate_1",
  "line": [512.5, 384.2, 10.0, 0.0],
  "mag": 12.5,
  "id": 0,
  "category_id": 0,
  "area": 1,
  "blend_perc": 0.0,
  "snr": 8.7,
  "type": "line"
}
```

## Configuration Options

### Export Settings
- `write_png`: Save PNG images (default: False)
- `write_fits`: Save FITS images (default: True)
- `save_annotated_images`: Save images with annotations (default: False)
- `remove_median`: Remove median from images (default: False)

### Detection Parameters
- `snr_cut`: Minimum SNR for annotations (default: 0.5)
- `box_size`: Bounding box size for point sources (default: 4)
- `streak_box_size`: Bounding box size for satellites (default: 10)
- `mask_radius`: Radius to mask around center (pixels, optional)

### Processing Options
- `apply_calibrations`: Apply calibrations from headers (default: True)

## Dataset Splitting

After exporting, you can split the dataset into train/val/test sets:

```python
from senpai.export.dataset_split import split_coco_dataset

split_coco_dataset(
    input_dir="/path/to/exported/data",
    output_dir="/path/to/split/dataset",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    random_seed=42
)
```

## Supported Data Types

### Sidereal Frames
- **Point sources**: Star detections with magnitude and SNR
- **Catalog stars**: Reference stars with known magnitudes

### Rate Track Frames
- **Satellites**: Point source detections of moving objects
- **Streaks**: Line annotations for star trails
- **Star detections**: Background star measurements

## Dependencies

- `numpy`: Array operations
- `astropy`: FITS file handling
- `png`: PNG image writing
- `senpai.engine`: Core SENPAI functionality

## Notes

- The exporter handles both live `SenpaiRun` objects and serialized `SenpaiRunResult` objects
- Star SNR is estimated from magnitude or counts when not directly available
- Annotated images require the `plot_single_frame` function from SENPAI's plotting module
- Individual COCO files are created per image to match the existing `rate_to_coco` pattern 