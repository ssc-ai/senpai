"""Dataset splitting functionality for COCO format exports."""

import json
import logging
import random
import shutil
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Represents a dataset split with train/val/test ratios."""

    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self):
        """Validate that ratios sum to 1.0."""
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class DatasetSplitter:
    """Split COCO datasets into train/val/test sets."""

    def __init__(self, split: DatasetSplit, random_seed: Optional[int] = None):
        """Initialize the dataset splitter.

        Args:
            split: Dataset split configuration
            random_seed: Random seed for reproducible splits
        """
        self.split = split
        if random_seed is not None:
            random.seed(random_seed)

    def split_coco_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        image_pattern: str = "*.fits",
        annotation_pattern: str = "*_point_sat.json",
        exclude_sidereal_from_lines: bool = True,  # Add parameter to control sidereal exclusion
        temporal_split: bool = True,  # Add parameter to control temporal vs random splitting
    ) -> Dict[str, List[str]]:
        """Split a COCO dataset into train/val/test sets.

        Args:
            input_dir: Directory containing COCO files
            output_dir: Output directory for split datasets
            image_pattern: Pattern to match image files
            annotation_pattern: Pattern to match annotation files
            exclude_sidereal_from_lines: Whether to exclude sidereal frames from lines dataset
            temporal_split: Whether to split temporally (True) or randomly (False)

        Returns:
            Dictionary mapping split names to lists of image IDs
        """
        # Find all annotation files - look for both point and line annotations
        point_annotation_files = list(input_dir.glob("*_point_sat.json"))
        line_annotation_files = list(input_dir.glob("*_line_star.json"))
        
        if not point_annotation_files and not line_annotation_files:
            raise ValueError(f"No annotation files found in {input_dir}")

        logger.info(f"Found {len(point_annotation_files)} point annotation files and {len(line_annotation_files)} line annotation files")
        logger.debug(f"Point annotation files: {[f.name for f in point_annotation_files[:3]]}...")
        logger.debug(f"Line annotation files: {[f.name for f in line_annotation_files[:3]]}...")

        # Load point annotations and images
        point_images = []
        point_annotations = []
        point_image_id_to_file = {}
        point_annotation_types = set()

        for annotation_file in point_annotation_files:
            try:
                with open(annotation_file, "r") as f:
                    data = json.load(f)

                # Collect images and their file paths
                for img in data.get("images", []):
                    img_id = img["id"]
                    if img_id not in point_image_id_to_file:
                        point_images.append(img)
                        point_image_id_to_file[img_id] = img["file_name"]

                # Collect annotations and track types
                for ann in data.get("annotations", []):
                    point_annotations.append(ann)
                    point_annotation_types.add(ann.get("type", "unknown"))

            except Exception as e:
                logger.warning(f"Failed to load point annotation file {annotation_file}: {e}")
                continue

        # Load line annotations and images
        line_images = []
        line_annotations = []
        line_image_id_to_file = {}
        line_annotation_types = set()

        for annotation_file in line_annotation_files:
            try:
                with open(annotation_file, "r") as f:
                    data = json.load(f)

                # Collect images and their file paths
                for img in data.get("images", []):
                    img_id = img["id"]
                    if img_id not in line_image_id_to_file:
                        line_images.append(img)
                        line_image_id_to_file[img_id] = img["file_name"]

                # Collect annotations and track types
                for ann in data.get("annotations", []):
                    line_annotations.append(ann)
                    line_annotation_types.add(ann.get("type", "unknown"))

            except Exception as e:
                logger.warning(f"Failed to load line annotation file {annotation_file}: {e}")
                continue

        logger.info(f"Point annotations: {len(point_annotations)} annotations of types {point_annotation_types}")
        logger.info(f"Line annotations: {len(line_annotations)} annotations of types {line_annotation_types}")

        # Combine all unique images for splitting
        all_images = []
        all_image_ids = set()
        
        for img in point_images + line_images:
            if img["id"] not in all_image_ids:
                all_images.append(img)
                all_image_ids.add(img["id"])

        if not all_images:
            raise ValueError("No valid images found in annotation files")

        # Sort images by datetime to ensure temporal order
        # Look for datetime in various possible fields
        def extract_datetime(img):
            # Try different possible datetime fields
            datetime_fields = ['datetime', 'date_obs', 'date', 'time', 'timestamp', 'mjd']
            for field in datetime_fields:
                if field in img:
                    return img[field]
            
            # If no datetime field found, try to extract from filename
            filename = img.get('file_name', '')
            # Common patterns: YYYYMMDD_HHMMSS, YYYY-MM-DD_HH:MM:SS, etc.
            
            # Try various datetime patterns in filename
            patterns = [
                r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
                r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})',  # YYYY-MM-DD_HH:MM:SS
                r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',  # YYYY-MM-DDTHH:MM:SS
                r'(\d{8}T\d{6})',  # YYYYMMDDTHHMMSS
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    dt_str = match.group(1)
                    try:
                        # Try different datetime formats
                        for fmt in ['%Y%m%d_%H%M%S', '%Y-%m-%d_%H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y%m%dT%H%M%S']:
                            try:
                                return datetime.strptime(dt_str, fmt)
                            except ValueError:
                                continue
                    except:
                        pass
            
            # If no datetime can be extracted, use a default value
            logger.warning(f"No datetime found for image {img.get('file_name', 'unknown')}, using default")
            return datetime.min

        if temporal_split:
            # Sort images by datetime
            try:
                all_images.sort(key=extract_datetime)
                logger.info("Sorted images by datetime for temporal splitting")
            except Exception as e:
                logger.warning(f"Failed to sort by datetime: {e}. Using original order.")
                # If sorting fails, keep original order but log warning

            # Extract image IDs in sorted order (no shuffling)
            image_ids = [img["id"] for img in all_images]
            logger.info(f"Found {len(image_ids)} unique images to split in temporal order")
        else:
            # Use random splitting (original behavior)
            image_ids = list(set(img["id"] for img in all_images))
            random.shuffle(image_ids)
            logger.info(f"Found {len(image_ids)} unique images to split randomly")

        # Calculate split sizes
        n_total = len(image_ids)
        n_train = int(n_total * self.split.train)
        n_val = int(n_total * self.split.val)

        train_ids = set(image_ids[:n_train])
        val_ids = set(image_ids[n_train : n_train + n_val])
        test_ids = set(image_ids[n_train + n_val :])

        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        split_type = "temporally" if temporal_split else "randomly"
        logger.info(f"Split {n_total} images {split_type}: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        
        # Log temporal range information if using temporal splitting
        if temporal_split and all_images:
            train_images = [img for img in all_images[:n_train]]
            val_images = [img for img in all_images[n_train:n_train + n_val]]
            test_images = [img for img in all_images[n_train + n_val:]]
            
            if train_images:
                train_start = extract_datetime(train_images[0])
                train_end = extract_datetime(train_images[-1])
                logger.info(f"Train set temporal range: {train_start} to {train_end}")
            
            if val_images:
                val_start = extract_datetime(val_images[0])
                val_end = extract_datetime(val_images[-1])
                logger.info(f"Val set temporal range: {val_start} to {val_end}")
            
            if test_images:
                test_start = extract_datetime(test_images[0])
                test_end = extract_datetime(test_images[-1])
                logger.info(f"Test set temporal range: {test_start} to {test_end}")

        # Create output directories
        for split_name in splits:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

        # Create annotations directory
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Process each split
        for split_name, split_image_ids in splits.items():
            logger.info(f"Processing {split_name} split with {len(split_image_ids)} images")

            # Filter images for this split
            split_images = [img for img in all_images if img["id"] in split_image_ids]
            
            # Filter point annotations for this split
            split_point_annotations = [ann for ann in point_annotations if ann["image_id"] in split_image_ids]
            
            # Filter line annotations for this split
            split_line_annotations = [ann for ann in line_annotations if ann["image_id"] in split_image_ids]

            # Copy image files to split directory (all images go to main directory)
            for img in split_images:
                img_filename = img["file_name"]
                src_path = input_dir / img_filename
                if src_path.exists():
                    dst_path = output_dir / split_name / src_path.name
                    shutil.copy2(src_path, dst_path)
                else:
                    logger.warning(f"Image file not found: {src_path}")

            # Create combined annotation files
            self._create_combined_annotations(split_images, split_point_annotations, split_line_annotations, annotations_dir, split_name, exclude_sidereal_from_lines)

        logger.info(f"Split datasets saved to {output_dir}")
        return {name: list(ids) for name, ids in splits.items()}

    def _create_combined_annotations(
        self, images: List[Dict], point_annotations: List[Dict], line_annotations: List[Dict], annotations_dir: Path, split_name: str, exclude_sidereal_from_lines: bool
    ):
        """Create combined annotation files for a split."""

        # Create points annotation file (satellites)
        points_data_dir = f"{split_name}/"
            
        points_data = {
            "info": {"data_dir": points_data_dir},
            "images": images,
            "annotations": point_annotations,
            "categories": [
                {
                    "id": 0,
                    "isthing": 1,
                    "supercategory": "point",
                    "name": "satellite",
                }
            ],
        }

        points_file = annotations_dir / f"points_{split_name}.json"
        with open(points_file, "w") as f:
            json.dump(points_data, f, indent=2)

        # Create lines annotation file (stars) - exclude sidereal frames
        # Sidereal frames don't have streak lines, only point sources
        rate_images = [img for img in images if img.get("type") == "rate"]
        rate_image_ids = {img["id"] for img in rate_images}
        
        lines_data_dir = f"{split_name}/"
        
        lines_data = {
            "info": {"data_dir": lines_data_dir},
            "images": rate_images,  # Only include rate frames
            "annotations": [ann for ann in line_annotations if ann.get("image_id") in rate_image_ids],  # Only line annotations from rate frames
            "categories": [
                {
                    "id": 0,
                    "isthing": 1,
                    "supercategory": "line",
                    "name": "star",
                }
            ],
        }

        lines_file = annotations_dir / f"lines_{split_name}.json"
        with open(lines_file, "w") as f:
            json.dump(lines_data, f, indent=2)

        logger.info(
            f"Created {split_name} annotations: {len(images)} images, "
            f"{len(points_data['annotations'])} point annotations, "
            f"{len(lines_data['annotations'])} line annotations"
        )

        # Add detailed debugging
        bbox_annotations = [ann for ann in point_annotations if ann.get("type") == "bbox"]
        line_annotations = [ann for ann in line_annotations if ann.get("type") == "line"]
        other_annotations = [ann for ann in point_annotations + line_annotations if ann.get("type") not in ["bbox", "line"]]
        
        # Count frame types
        sidereal_images = [img for img in images if img.get("type") == "sidereal"]
        rate_images = [img for img in images if img.get("type") == "rate"]
        
        logger.debug(f"{split_name} split details:")
        logger.debug(f"  - Images: {len(images)} total ({len(sidereal_images)} sidereal, {len(rate_images)} rate)")
        logger.debug(f"  - Bbox annotations: {len(bbox_annotations)}")
        logger.debug(f"  - Line annotations: {len(line_annotations)}")
        logger.debug(f"  - Other annotations: {len(other_annotations)}")
        if other_annotations:
            logger.debug(f"  - Other types: {set(ann.get('type') for ann in other_annotations)}")


def split_coco_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None,
    image_pattern: str = "*.fits",
    exclude_sidereal_from_lines: bool = True,
    temporal_split: bool = True,
) -> Dict[str, List[str]]:
    """Convenience function to split a COCO dataset.

    Args:
        input_dir: Directory containing COCO files
        output_dir: Output directory for split datasets
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducible splits
        image_pattern: Pattern to match image files
        exclude_sidereal_from_lines: Whether to exclude sidereal frames from lines dataset
        temporal_split: Whether to split temporally (True) or randomly (False)

    Returns:
        Dictionary mapping split names to lists of image IDs
    """
    split = DatasetSplit(train=train_ratio, val=val_ratio, test=test_ratio)
    splitter = DatasetSplitter(split, random_seed=random_seed)

    return splitter.split_coco_dataset(
        Path(input_dir),
        Path(output_dir),
        image_pattern=image_pattern,
        exclude_sidereal_from_lines=exclude_sidereal_from_lines,
        temporal_split=temporal_split,
    )
