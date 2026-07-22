"""Image masking utilities for locating and removing streaks in detection frames."""

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, maximum_filter


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

    Performs a flood fill from ``start_point`` across all pixels whose flux exceeds
    ``flux_threshold``, optionally dilating the resulting region.

    Args:
        image (np.ndarray): Image to flood fill over.
        start_point (tuple[int, int]): ``(row, column)`` seed pixel for the fill.
        flux_threshold (float): Pixels at or below this flux are treated as
            background and stop the fill.
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
    struct = generate_binary_structure(2, 2)  # 2D connectivity, diagonal neighbors included

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


def remove_streak_at_point(
    image: np.ndarray, start_point: tuple[int, int], fill_min: float, fill_mode: np.ufunc = np.mean
) -> np.ndarray:
    """Remove the streak/cluster containing a given point by overwriting its pixels.

    Args:
        image (np.ndarray): Image to modify in place.
        start_point (tuple[int, int]): ``(row, column)`` seed pixel of the streak.
        fill_min (float): Flux threshold used to delineate the streak cluster.
        fill_mode (np.ufunc): Reduction applied to ``image`` to compute the fill
            value for the masked pixels. Defaults to ``np.mean``.

    Returns:
        np.ndarray: The modified image with the streak pixels replaced.
    """
    mapped = map_cluster(image, start_point, fill_min)
    image[np.where(mapped)] = fill_mode(image)
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


def remove_near_saturation_streaks(image: np.ndarray, data_type: str) -> tuple[np.ndarray, int]:
    """Remove streaks near saturation.

    Returns:
        image: The image with streaks removed.
        removed_streak: The number of streaks removed.
    """
    # Use rate_frame's data type instead of hardcoded uint16
    max_val = 2 ** (np.dtype(data_type).itemsize * 8) - 1

    fill_min = np.median(image) + 0.4 * np.std(image)

    # remove any whopper streaks... they mess with stuff when close to saturation
    removed_streaks = 0
    filter_value = 0.90 * max_val

    while np.max(image) > filter_value:
        image = remove_brightest_streak(image, fill_min)
        removed_streaks += 1

    return image, removed_streaks


def remove_border_crossing_streaks(image: np.ndarray) -> np.ndarray:
    """Remove streaks that cross or touch the image border.

    Repeatedly removes the brightest streak anchored in the 2-pixel border region
    until no border pixel exceeds the brightness cut.

    Args:
        image (np.ndarray): Image to modify in place.

    Returns:
        np.ndarray: The modified image with border-crossing streaks removed.
    """
    # now we need to remove edge targets from rate frame...
    border_pixels = mask_all_but_border(image, 2)
    pixel_cut = np.mean(image) + 3 * np.std(image)
    fill_min = np.median(image) + 0.5 * np.std(image)

    while np.max(border_pixels) > pixel_cut:
        start_point = np.unravel_index(np.argmax(border_pixels), image.shape)
        image = remove_streak_at_point(image, start_point, fill_min)
        border_pixels = mask_all_but_border(image, 2)

    return image


def map_cluster_with_peaks(
    image: np.ndarray,
    start_point: tuple[int, int],
    flux_threshold: float,
    pad_size: int = 0,
    min_separation: int = 5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Map a cluster and identify multiple peaks within it.

    Args:
        image (np.ndarray): Image to flood fill over.
        start_point (tuple[int, int]): ``(row, column)`` seed pixel for the fill.
        flux_threshold (float): Pixels at or below this flux are treated as background.
        pad_size (int): Number of binary-dilation iterations to grow the cluster by.
            Defaults to 0.
        min_separation (int): Minimum pixel separation between detected peaks; sets
            the maximum-filter window size. Defaults to 5.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: The boolean cluster mask and a list of
            ``(row, column)`` peak coordinates within the cluster, sorted from
            brightest to faintest.
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
    peak_coords = sorted(peak_coords, key=lambda p: masked_image[p[0], p[1]], reverse=True)

    return cluster_mask, peak_coords
