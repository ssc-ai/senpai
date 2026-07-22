"""Preserved v2.6 image masking utilities for locating and removing streaks.

Extended ("extra") variant of :mod:`senpai.engine.detection.streak.masking`, adding
bounded flood fills, robust connected-component streak removal, FWHM-based shape
analysis, and pairwise border-crossing removal for correlation frame pairs.
"""

import logging

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
    label,
    maximum_filter,
)


def percent_difference(a: float, b: float) -> float:
    """Compute the percent difference between two values relative to their mean.

    Args:
        a (float): First value.
        b (float): Second value.

    Returns:
        float: The absolute difference as a percentage of the mean of ``a`` and
            ``b``. Returns ``0.0`` when both values are zero.
    """
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / ((a + b) / 2) * 100


def mask_tol(img: np.ndarray, center: tuple[int, int], pixel_tol: int = 30) -> np.ndarray:
    """Build a circular mask centered on a point within an image.

    Args:
        img (np.ndarray): Image whose shape determines the mask dimensions.
        center (tuple[int, int]): ``(row, column)`` center of the circular region.
        pixel_tol (int): Radius of the circular region in pixels. Defaults to 30.

    Returns:
        np.ndarray: An array matching ``img`` shape with ``1`` inside the circle
            (distance from ``center`` <= ``pixel_tol``) and ``0`` elsewhere.
    """
    mask = np.zeros(shape=img.shape)
    radius = pixel_tol

    # Create coordinate grids
    y, x = np.ogrid[: img.shape[0], : img.shape[1]]

    # Calculate squared distance from center
    dist_squared = (y - center[0]) ** 2 + (x - center[1]) ** 2

    # Set mask to 1 where distance <= radius
    mask[dist_squared <= radius**2] = 1

    return mask


def map_cluster(
    image: np.ndarray,
    start_point: tuple[int, int],
    flux_threshold: float,
    pad_size: int = 0,
) -> np.ndarray:
    """Map a cluster starting from a given point until a specified flux threshold is met.

    Args:
        image (np.ndarray): Image to flood fill over.
        start_point (tuple[int, int]): ``(row, column)`` seed pixel for the fill.
        flux_threshold (float): Pixels at or below this flux stop the fill.
        pad_size (int): Number of binary-dilation iterations to grow the cluster by.
            Defaults to 0 (no dilation).

    Returns:
        np.ndarray: A boolean mask of the connected cluster, dilated by ``pad_size``
            when nonzero. An empty mask is returned if ``start_point`` lies outside
            the image bounds.
    """
    # Create a binary mask where true values are below the flux threshold
    threshold_mask = image <= flux_threshold

    # Define a connectivity structure that considers neighbors in all directions
    struct = generate_binary_structure(
        2, 2
    )  # 2D connectivity, diagonal neighbors included

    # Create an array of zeros
    visited = np.zeros_like(image, dtype=bool)

    # Start flood fill, unless start point outside
    if start_point[0] >= image.shape[0] or start_point[1] >= image.shape[1]:
        return visited

    stack = [start_point]

    while stack:
        x, y = stack.pop()
        if not visited[x, y] and not threshold_mask[x, y]:
            visited[x, y] = True
            # Push the neighboring pixels onto the stack
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                        stack.append((nx, ny))

    if pad_size == 0:
        return visited

    padded_cluster = binary_dilation(visited, structure=struct, iterations=pad_size)

    return padded_cluster


def map_cluster_bounded(
    image: np.ndarray,
    start_point: tuple[int, int],
    flux_threshold: float,
    max_radius: int = 500,
    max_pixels: int = 10000,
    pad_size: int = 0,
) -> np.ndarray:
    """Map a cluster with hard limits on radius and pixel count to prevent runaway flood fills.

    Args:
        image: The image data
        start_point: (y, x) starting point for flood fill
        flux_threshold: Pixels above this value are included in the cluster
        max_radius: Maximum distance from start_point to explore (in pixels)
        max_pixels: Maximum number of pixels to include (stops early if exceeded)
        pad_size: Optional padding to apply after flood fill

    Returns:
        Boolean mask of the cluster
    """
    # Create a binary mask where true values are below the flux threshold
    threshold_mask = image <= flux_threshold

    # Define a connectivity structure that considers neighbors in all directions
    struct = generate_binary_structure(
        2, 2
    )  # 2D connectivity, diagonal neighbors included

    # Create an array of zeros
    visited = np.zeros_like(image, dtype=bool)

    # Start flood fill, unless start point outside
    if start_point[0] >= image.shape[0] or start_point[1] >= image.shape[1]:
        return visited

    # Calculate radius bounds
    start_y, start_x = start_point
    y_min = max(0, start_y - max_radius)
    y_max = min(image.shape[0], start_y + max_radius + 1)
    x_min = max(0, start_x - max_radius)
    x_max = min(image.shape[1], start_x + max_radius + 1)

    stack = [start_point]
    pixel_count = 0

    while stack and pixel_count < max_pixels:
        y, x = stack.pop()

        # Check if already visited or outside radius bounds
        if visited[y, x]:
            continue
        if not (y_min <= y < y_max and x_min <= x < x_max):
            continue
        if threshold_mask[y, x]:
            continue

        visited[y, x] = True
        pixel_count += 1

        # Push the neighboring pixels onto the stack
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1] and not visited[ny, nx]:
                    stack.append((ny, nx))

    if pad_size == 0:
        return visited

    padded_cluster = binary_dilation(visited, structure=struct, iterations=pad_size)

    return padded_cluster


def analyze_source_shape_fwhm(
    image: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    weights: np.ndarray = None,
) -> dict:
    """Analyze the shape of a source using FWHM-based thresholding for robust measurements.

    Args:
        image: The image data
        y_coords: Y coordinates of the source pixels
        x_coords: X coordinates of the source pixels
        weights: Optional weights for each pixel (if None, uses image values)

    Returns:
        dict: Dictionary containing shape analysis results including:
            - center: (y, x) centroid position
            - length: Length measurement based on FWHM thresholding
            - orientation: Orientation in degrees
            - fwhm_major: FWHM along major axis
            - fwhm_minor: FWHM along minor axis
            - fwhm_threshold: Threshold used for FWHM calculation
            - fwhm_pixels: Number of pixels used in FWHM analysis
            - total_pixels: Total number of input pixels
    """
    if len(y_coords) == 0:
        return {
            "center": (0, 0),
            "length": 0.0,
            "orientation": 0.0,
            "fwhm_major": 0.0,
            "fwhm_minor": 0.0,
            "fwhm_threshold": 0.0,
            "fwhm_pixels": 0,
            "total_pixels": 0,
        }

    # Use image values as weights if not provided
    if weights is None:
        weights = image[y_coords, x_coords]

    # Calculate initial weighted centroid using all points
    total_weight = np.sum(weights)
    if total_weight > 0:
        centroid_y = np.sum(y_coords * weights) / total_weight
        centroid_x = np.sum(x_coords * weights) / total_weight
        centroid = (centroid_y, centroid_x)
    else:
        centroid = (np.mean(y_coords), np.mean(x_coords))

    # Get image values at the coordinates
    mapped_values = image[y_coords, x_coords]
    peak_value = np.max(mapped_values)

    # Estimate background within the mapped region
    # Use values below 50th percentile as background estimate
    background_values = mapped_values[mapped_values <= np.percentile(mapped_values, 50)]
    background_level = np.median(background_values) if len(background_values) > 0 else np.min(mapped_values)

    # Calculate FWHM threshold: halfway between background and peak
    fwhm_threshold = background_level + 0.5 * (peak_value - background_level)

    # Get pixels above the FWHM threshold for shape analysis
    fwhm_mask = mapped_values >= fwhm_threshold
    fwhm_y_coords = y_coords[fwhm_mask]
    fwhm_x_coords = x_coords[fwhm_mask]
    fwhm_weights = mapped_values[fwhm_mask]

    # Use FWHM-thresholded points for shape analysis if we have enough points
    if len(fwhm_y_coords) >= 3:
        # Use FWHM-thresholded points for calculation
        analysis_y_coords = fwhm_y_coords
        analysis_x_coords = fwhm_x_coords
        analysis_weights = fwhm_weights

        # Recalculate centroid using FWHM-thresholded points
        total_fwhm_weight = np.sum(analysis_weights)
        if total_fwhm_weight > 0:
            centroid_y = (
                np.sum(analysis_y_coords * analysis_weights) / total_fwhm_weight
            )
            centroid_x = (
                np.sum(analysis_x_coords * analysis_weights) / total_fwhm_weight
            )
            centroid = (centroid_y, centroid_x)
    else:
        # Fall back to using all points if FWHM thresholding leaves too few points
        analysis_y_coords = y_coords
        analysis_x_coords = x_coords
        analysis_weights = weights

    # Calculate covariance matrix for orientation using analysis points
    y_diff = analysis_y_coords - centroid[0]
    x_diff = analysis_x_coords - centroid[1]

    total_analysis_weight = np.sum(analysis_weights)
    if total_analysis_weight > 0:
        cov_xx = np.sum(analysis_weights * x_diff * x_diff) / total_analysis_weight
        cov_yy = np.sum(analysis_weights * y_diff * y_diff) / total_analysis_weight
        cov_xy = np.sum(analysis_weights * x_diff * y_diff) / total_analysis_weight
    else:
        cov_xx = cov_yy = cov_xy = 0

    # Calculate orientation (angle in radians)
    if cov_xx == cov_yy:
        # Handle the case where the covariance matrix is isotropic
        orientation = 0.0 if cov_xy == 0 else np.pi / 4.0
    else:
        orientation = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)

    # Convert to degrees
    orientation_deg = np.degrees(orientation)

    # Calculate length using principal component analysis on analysis points
    # Eigenvalues of the covariance matrix give the variance along the principal axes
    try:
        evals, evecs = np.linalg.eig(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]]))

        # Calculate FWHM (Full Width at Half Maximum) using FWHM-thresholded points
        # For a Gaussian distribution, FWHM = 2.355 * sigma
        # Where sigma is the standard deviation (sqrt of eigenvalue)
        fwhm_major = (
            2.355 * np.sqrt(np.max(evals))
            if len(evals) > 0 and np.max(evals) > 0
            else 0
        )
        fwhm_minor = (
            2.355 * np.sqrt(np.min(evals))
            if len(evals) > 0 and np.min(evals) > 0
            else 0
        )

        # For length measurement, use ALL pixels (not just FWHM-thresholded ones)
        # Project all pixels onto the principal axis and measure the full extent
        # This ensures we capture the entire streak length, not just the bright portion
        if len(evals) > 0 and np.max(evals) > 0:
            # Get the eigenvector corresponding to the largest eigenvalue (major axis)
            major_evec = evecs[:, np.argmax(evals)]

            # Project all pixels onto the principal axis
            # Center all pixels relative to the centroid
            y_centered = y_coords - centroid[0]
            x_centered = x_coords - centroid[1]

            # Project each point onto the major axis using the eigenvector
            # The eigenvector is [evec_x, evec_y], so projection is dot product
            projections = x_centered * major_evec[0] + y_centered * major_evec[1]

            # Length is the full extent along the principal axis
            length = (
                np.max(projections) - np.min(projections) if len(projections) > 0 else 0
            )
        else:
            length = 0

    except (np.linalg.LinAlgError, ValueError):
        # If eigenvalue decomposition fails, use simple estimates with ALL pixels
        length = np.sqrt(
            (np.max(x_coords) - np.min(x_coords)) ** 2
            + (np.max(y_coords) - np.min(y_coords)) ** 2
        )
        fwhm_major = length / 4.0
        fwhm_minor = fwhm_major / 2.0

    return {
        "center": centroid,
        "length": length,
        "orientation": orientation_deg,
        "fwhm_major": fwhm_major,
        "fwhm_minor": fwhm_minor,
        "fwhm_threshold": fwhm_threshold,
        "fwhm_pixels": len(fwhm_y_coords),
        "total_pixels": len(y_coords),
    }


def remove_streak_at_point_robust(
    image: np.ndarray,
    start_point: tuple[int, int],
    box_size: int,
    fill_mode: np.ufunc = np.mean,
    thresholds: list[float] | None = None,
    pad_size: int = 2,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, dict]:
    """Remove a streak using robust connected component analysis.

    This approach is more robust than threshold-based flood fill when dealing with
    high signal or high variance regions, as it:
    1. Normalizes the local region to 0-1 range
    2. Uses multiple thresholds to find contiguous regions
    3. Only removes the largest connected component at each threshold
    4. Optionally dilates the mask to ensure complete removal

    Args:
        image: The image data
        start_point: (y, x) center point of the streak (row, col)
        box_size: Size of the local region to extract around the bright point
        fill_mode: Function to use for filling (default np.mean)
        thresholds: List of intensity thresholds (0-1 range) to try.
                   Defaults to [0.15, 0.25, 0.35, 0.45] (lower = more aggressive)
        pad_size: Number of dilation iterations to expand the mask (default 2)
        logger: Optional logger for debug output

    Returns:
        tuple: (modified image, info dict with removal statistics)
    """
    if thresholds is None:
        thresholds = [
            0.15,
            0.25,
            0.35,
            0.45,
        ]  # Lower thresholds for more aggressive removal

    y_center, x_center = start_point

    if logger:
        logger.debug(
            f"remove_streak_at_point_robust: start_point={start_point}, "
            f"unpacked as y_center={y_center}, x_center={x_center}, "
            f"image.shape={image.shape}"
        )

    # Extract local region around the bright point
    y_min = max(0, y_center - box_size)
    y_max = min(image.shape[0], y_center + box_size)
    x_min = max(0, x_center - box_size)
    x_max = min(image.shape[1], x_center + box_size)

    local_region = image[y_min:y_max, x_min:x_max].copy()

    if local_region.size == 0:
        return image, {"num_pixels": 0, "thresholds_tried": 0}

    # Normalize to 0-1 range
    region_min = np.min(local_region)
    region_max = np.max(local_region)

    if region_max <= region_min:
        return image, {"num_pixels": 0, "thresholds_tried": 0}

    normalized_region = (local_region - region_min) / (region_max - region_min)

    # Calculate the center point in local coordinates
    local_y = y_center - y_min
    local_x = x_center - x_min

    # Ensure center point is within bounds
    if (
        local_y < 0
        or local_y >= local_region.shape[0]
        or local_x < 0
        or local_x >= local_region.shape[1]
    ):
        if logger:
            logger.warning(
                f"Center point ({local_y}, {local_x}) outside local region bounds"
            )
        return image, {"num_pixels": 0, "thresholds_tried": 0}

    # Accumulate mask across all thresholds
    combined_mask = np.zeros_like(normalized_region, dtype=bool)
    pixels_removed_per_threshold = []

    # Try each threshold and find the component that CONTAINS the center point
    for thresh in thresholds:
        # Find pixels above threshold
        thresh_mask = normalized_region > thresh

        if not np.any(thresh_mask):
            pixels_removed_per_threshold.append(0)
            continue

        # Check if center point is above threshold at this level
        if not thresh_mask[local_y, local_x]:
            pixels_removed_per_threshold.append(0)
            continue

        # Use connected component analysis
        labeled_mask, num_features = label(thresh_mask)

        if num_features == 0:
            pixels_removed_per_threshold.append(0)
            continue

        # Find the component that contains the center point
        center_component = labeled_mask[local_y, local_x]

        if center_component == 0:
            # Center point is not in any component (shouldn't happen given check above)
            pixels_removed_per_threshold.append(0)
            continue

        center_mask = labeled_mask == center_component

        # Add to combined mask
        combined_mask |= center_mask
        pixels_removed_per_threshold.append(np.sum(center_mask))

    # Dilate the mask to ensure complete removal of streak edges
    if pad_size > 0 and np.any(combined_mask):
        struct = generate_binary_structure(2, 2)  # 8-connectivity
        combined_mask = binary_dilation(
            combined_mask, structure=struct, iterations=pad_size
        )

    # Map the local mask back to the full image coordinates
    full_mask = np.zeros_like(image, dtype=bool)
    full_mask[y_min:y_max, x_min:x_max] = combined_mask

    # Get fill value
    fill_value = fill_mode(image)

    # Fill the masked region
    image[full_mask] = fill_value

    info = {
        "num_pixels": int(np.sum(full_mask)),  # Total pixels after dilation
        "num_pixels_before_dilation": int(np.sum(pixels_removed_per_threshold)),
        "thresholds_tried": len(thresholds),
        "pixels_per_threshold": pixels_removed_per_threshold,
        "region_min": float(region_min),
        "region_max": float(region_max),
        "fill_value": float(fill_value),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_center": int(y_center),
        "x_center": int(x_center),
        "pad_size": pad_size,
    }

    if logger:
        logger.debug(
            f"Removed region: y[{y_min}:{y_max}], x[{x_min}:{x_max}], "
            f"center=({y_center}, {x_center}), pixels={np.sum(full_mask)} "
            f"(before dilation: {np.sum(pixels_removed_per_threshold)})"
        )

    return image, info


def remove_streak_at_point_enriched(
    image: np.ndarray,
    start_point: tuple[int, int],
    fill_min: float,
    fill_mode: np.ufunc = np.mean,
    max_radius: int | None = None,
    max_pixels: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Remove the streak containing a point and report its measured geometry.

    Bounds a flood fill around ``start_point``, fills the mapped cluster with the
    reduction of ``image``, and runs an FWHM-based shape analysis on the region.

    Args:
        image (np.ndarray): Image to modify in place.
        start_point (tuple[int, int]): ``(row, column)`` seed pixel of the streak.
        fill_min (float): Flux threshold used to delineate the streak cluster.
        fill_mode (np.ufunc): Reduction applied to ``image`` to compute the fill value.
            Defaults to ``np.mean``.
        max_radius (int): Maximum flood-fill radius in pixels. Defaults to 2000 when None.
        max_pixels (int): Maximum flood-fill pixel count. Defaults to 200000 when None.

    Returns:
        tuple[np.ndarray, dict]: The modified image and a dict of streak properties
            (``length``, ``orientation``, ``center``, ``num_pixels``, ``fwhm_major``,
            ``fwhm_minor``, ``fwhm_threshold``, ``fwhm_pixels``).
    """
    # Always bound the flood fill. An unbounded fill runs away across the whole
    # frame whenever a bright feature connects large regions above the threshold
    # — e.g. a dead row/column, or overlapping streaks in a crowded field — and
    # map_cluster's stack also inflates ~8x because it pushes neighbors without a
    # pre-push visited check. map_cluster_bounded caps both and pre-checks on
    # push. Defaults are generous so real streaks are never clipped.
    mapped = map_cluster_bounded(
        image,
        start_point,
        fill_min,
        max_radius=max_radius if max_radius is not None else 2000,
        max_pixels=max_pixels if max_pixels is not None else 200000,
    )

    # Get coordinates of all points in the streak
    y_coords, x_coords = np.where(mapped)

    # If no points were mapped, return early
    if len(y_coords) == 0:
        return image, {"length": 0, "orientation": 0, "center": start_point}

    # Use the shared FWHM-based analysis function
    analysis_result = analyze_source_shape_fwhm(image, y_coords, x_coords)

    # Fill the streak with the specified fill mode (using original mapped region)
    image[np.where(mapped)] = fill_mode(image)

    # Return the modified image and streak properties
    streak_info = {
        "length": analysis_result["length"],
        "orientation": analysis_result["orientation"],
        "center": analysis_result["center"],
        "num_pixels": analysis_result["total_pixels"],  # Total mapped pixels
        "fwhm_major": analysis_result["fwhm_major"],
        "fwhm_minor": analysis_result["fwhm_minor"],
        "fwhm_threshold": analysis_result["fwhm_threshold"],  # Store for debugging
        "fwhm_pixels": analysis_result[
            "fwhm_pixels"
        ],  # Number of pixels used for length calc
    }

    return image, streak_info


def remove_streak_at_point(
    image: np.ndarray,
    start_point: tuple[int, int],
    fill_min: float,
    fill_mode: np.ufunc = np.mean,
    max_radius: int | None = None,
    max_pixels: int | None = None,
) -> np.ndarray:
    """Remove the streak/cluster containing a given point by overwriting its pixels.

    Args:
        image (np.ndarray): Image to modify in place.
        start_point (tuple[int, int]): ``(row, column)`` seed pixel of the streak.
        fill_min (float): Flux threshold used to delineate the streak cluster.
        fill_mode (np.ufunc): Reduction applied to ``image`` to compute the fill value.
            Defaults to ``np.mean``.
        max_radius (int): Maximum flood-fill radius in pixels, forwarded to the
            enriched routine.
        max_pixels (int): Maximum flood-fill pixel count, forwarded to the enriched
            routine.

    Returns:
        np.ndarray: The modified image with the streak pixels replaced.
    """
    image, _ = remove_streak_at_point_enriched(
        image, start_point, fill_min, fill_mode, max_radius, max_pixels
    )
    return image


def remove_brightest_streak(image: np.ndarray, fill_min: float) -> np.ndarray:
    """Remove the streak containing the image's brightest pixel.

    Args:
        image (np.ndarray): Image to modify in place.
        fill_min (float): Flux threshold used to delineate the streak cluster.

    Returns:
        np.ndarray: The modified image with the brightest streak removed.
    """
    start_point = np.unravel_index(np.argmax(image), image.shape)
    return remove_streak_at_point(image, start_point, fill_min)


def mask_all_but_border(image: np.ndarray, n_pixels: int = 1) -> np.ndarray:
    """Zero out the interior of an image, keeping only an outer border.

    Args:
        image (np.ndarray): Image to copy and mask.
        n_pixels (int): Width in pixels of the border to retain. Defaults to 1.

    Returns:
        np.ndarray: A copy of ``image`` with all but the outer ``n_pixels`` border set to zero.
    """
    border_pixels = image.copy()
    border_pixels[n_pixels:-n_pixels, n_pixels:-n_pixels] = 0.0
    return border_pixels


def mask_border(image: np.ndarray, n_pixels: int = 1) -> np.ndarray:
    """Zero out an outer border of an image, keeping the interior.

    Args:
        image (np.ndarray): Image to copy and mask.
        n_pixels (int): Width in pixels of the border to zero out. Defaults to 1.

    Returns:
        np.ndarray: A copy of ``image`` with the outer ``n_pixels`` border set to zero.
    """
    pixels = image.copy()
    pixels[0:n_pixels, :] = 0.0
    pixels[-n_pixels:, :] = 0.0
    pixels[:, 0:n_pixels] = 0.0
    pixels[:, -n_pixels:] = 0.0
    return pixels


def remove_n_brightest_streaks(image: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Remove the ``n`` brightest streaks from an image.

    Args:
        image (np.ndarray): Image to modify in place.
        n (int): Number of brightest streaks to remove.

    Returns:
        tuple[np.ndarray, int]: The modified image and the number of streaks removed.
    """
    removed_streaks = 0
    fill_min = np.median(image) + 0.5 * np.std(image)

    for _ in range(n):
        image = remove_brightest_streak(image, fill_min)
        removed_streaks += 1

    return image, removed_streaks


def remove_near_saturation_streaks(
    image: np.ndarray, data_type: str
) -> tuple[np.ndarray, int]:
    """Remove streaks near saturation.

    Vectorized: label every connected blob above ``fill_min`` once, then fill
    the blobs that contain a near-saturated pixel — all in a single pass. The
    previous implementation removed one blob per ``while`` iteration via a
    full-frame ``argmax`` + flood fill, which is O(n_saturated_sources x
    frame_size); on a crowded 66 MP rate frame with ~1000 saturated cores that
    cost minutes (and effectively hung). This is O(frame_size) with the same
    result.

    Returns:
        image: The image with streaks removed.
        removed_streak: The number of streaks removed.
    """
    # Use rate_frame's data type instead of hardcoded uint16
    max_val = 2 ** (np.dtype(data_type).itemsize * 8) - 1
    filter_value = 0.90 * max_val

    if np.max(image) <= filter_value:
        return image, 0

    fill_min = np.median(image) + 0.4 * np.std(image)
    fill_value = float(np.mean(image))

    struct = generate_binary_structure(2, 2)
    labels, _ = label(image > fill_min, structure=struct)
    # Components that contain at least one near-saturated pixel.
    sat_labels = np.unique(labels[image > filter_value])
    sat_labels = sat_labels[sat_labels != 0]
    if sat_labels.size == 0:
        return image, 0

    image[np.isin(labels, sat_labels)] = fill_value
    return image, int(sat_labels.size)


def _border_crossing_mask(image: np.ndarray) -> np.ndarray | None:
    """Mask of streaks crossing the frame border (None when there are none).

    Vectorized for the same reason as remove_near_saturation_streaks: label
    blobs above fill_min once, then select the ones seeded by a bright
    (> pixel_cut) pixel in the 2px border, rather than looping a full-frame
    argmax + mask copy per streak.
    """
    pixel_cut = np.mean(image) + 3 * np.std(image)
    fill_min = np.median(image) + 0.5 * np.std(image)

    border_seed = np.zeros(image.shape, dtype=bool)
    border_seed[:2, :] = True
    border_seed[-2:, :] = True
    border_seed[:, :2] = True
    border_seed[:, -2:] = True
    border_seed &= image > pixel_cut
    if not border_seed.any():
        return None

    struct = generate_binary_structure(2, 2)
    labels, _ = label(image > fill_min, structure=struct)
    seed_labels = np.unique(labels[border_seed])
    seed_labels = seed_labels[seed_labels != 0]
    if not seed_labels.size:
        return None
    return np.isin(labels, seed_labels)


def remove_border_crossing_streaks(image: np.ndarray) -> np.ndarray:
    """Remove streaks that cross or touch the image border.

    Args:
        image (np.ndarray): Image to modify in place.

    Returns:
        np.ndarray: The modified image with border-crossing streaks filled to the mean.
    """
    # Remove edge targets (streaks crossing the frame border).
    mask = _border_crossing_mask(image)
    if mask is not None:
        image[mask] = float(np.mean(image))
    return image


def _translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a boolean mask by (dx, dy) pixels with zero fill (no wraparound)."""
    dx, dy = int(dx), int(dy)
    out = np.zeros_like(mask)
    h, w = mask.shape
    # A shift beyond the frame moves everything off-image (multi-frame-gap
    # hops can have expected drifts larger than the frame itself) — and the
    # slice arithmetic below would wrap a negative stop instead of clamping.
    if abs(dx) >= w or abs(dy) >= h:
        return out
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    yt0, yt1 = max(0, -dy), min(h, h - dy)
    xt0, xt1 = max(0, -dx), min(w, w - dx)
    out[ys0:ys1, xs0:xs1] = mask[yt0:yt1, xt0:xt1]
    return out


def remove_border_crossing_streaks_pairwise(
    image_a: np.ndarray,
    image_b: np.ndarray,
    drift_dx: float,
    drift_dy: float,
    pad_px: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Remove border-crossing streaks from a frame pair, symmetrically.

    Deleting a streak from only one frame of a correlation pair breaks its
    true match while its counterpart (drifted ~one hop inward) survives in
    the other frame — in sparse fields the whitened cross-correlation then
    locks onto a mis-pair of two *different* streaks, producing a reversed or
    aliased shift. So whenever a blob is filled in one frame, also fill the
    counterpart region in the other frame: the blob mask translated by the
    expected inter-frame drift. The drift sign is unknown here (streak axes
    carry a 180° ambiguity), so both ±drift translations are filled; the
    spurious one costs a blob-sized patch of sky, which is harmless next to a
    broken pair.

    Args:
        image_a: The first frame (modified in place, same shape as ``image_b``).
        image_b: The second frame (modified in place, same shape as ``image_a``).
        drift_dx: Expected inter-frame drift along x (sign-agnostic).
        drift_dy: Expected inter-frame drift along y (sign-agnostic).
        pad_px: Isotropic dilation of the counterpart fills, absorbing drift
            estimate error.

    Returns:
        A tuple ``(image_a, image_b, filled_px_a, filled_px_b)`` of the two modified
        frames and the number of pixels filled in each.
    """
    from scipy.ndimage import binary_dilation

    mask_a = _border_crossing_mask(image_a)
    mask_b = _border_crossing_mask(image_b)

    def counterpart(mask: np.ndarray) -> np.ndarray:
        both = _translate_mask(mask, round(drift_dx), round(drift_dy)) | _translate_mask(
            mask, -round(drift_dx), -round(drift_dy)
        )
        if pad_px > 0:
            both = binary_dilation(both, iterations=pad_px)
        return both

    fill_a = mask_a.copy() if mask_a is not None else np.zeros(image_a.shape, dtype=bool)
    fill_b = mask_b.copy() if mask_b is not None else np.zeros(image_b.shape, dtype=bool)
    if mask_b is not None:
        fill_a |= counterpart(mask_b)
    if mask_a is not None:
        fill_b |= counterpart(mask_a)

    if fill_a.any():
        image_a[fill_a] = float(np.mean(image_a))
    if fill_b.any():
        image_b[fill_b] = float(np.mean(image_b))
    return image_a, image_b, int(fill_a.sum()), int(fill_b.sum())


def map_cluster_with_peaks(
    image: np.ndarray,
    start_point: tuple[int, int],
    flux_threshold: float,
    pad_size: int = 0,
    min_separation: int = 5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Map a cluster and identify multiple peaks within it.

    Args:
        image: Image to flood fill over.
        start_point: ``(row, column)`` seed pixel for the fill.
        flux_threshold: Pixels at or below this flux are treated as background.
        pad_size: Number of binary-dilation iterations to grow the cluster by.
            Defaults to 0.
        min_separation: Minimum pixel separation between detected peaks; sets the
            maximum-filter window size. Defaults to 5.

    Returns:
        A tuple of the boolean cluster mask and a list of ``(row, column)`` peak
        coordinates within the cluster, sorted from brightest to faintest.
    """
    # First map the cluster as before
    cluster_mask = map_cluster(image, start_point, flux_threshold, pad_size)

    # Create a masked version of the image
    masked_image = image.copy()
    masked_image[~cluster_mask] = 0

    # Find local maxima within the cluster

    # Apply maximum filter
    size = 2 * min_separation + 1
    max_filtered = maximum_filter(masked_image, size=size, mode="constant")

    # Find points that are local maxima
    maxima = (masked_image == max_filtered) & (masked_image > 0)

    # Get coordinates of maxima
    peak_coords = np.argwhere(maxima)

    # Sort by intensity (brightest first)
    peak_coords = sorted(
        peak_coords, key=lambda p: masked_image[p[0], p[1]], reverse=True
    )

    return cluster_mask, peak_coords
