"""Frame rendering: sky-gridded overlays, SIP-distortion maps, and shift/limit diagnostics."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

matplotlib.use("Svg")
import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import senpai
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.starfield import StarField, StarListImage
from senpai.engine.plotting.axes import prep_axes
from senpai.engine.plotting.normalization import zscale
from senpai.settings import settings

if TYPE_CHECKING:
    from photutils.aperture import Aperture

logger = logging.getLogger(__name__)


def plot_overs(
    ax: plt.Axes,
    starfield: StarField | None = None,
    starlist: StarListImage | None = None,
    detections: StarListImage | None = None,
    streak: StreakMetadata | None = None,
    streak_candidates: list | None = None,
    centercross: bool = True,
    marker: str | None = "+",
    markersize: float = 5,
    linewidth: float = 1,
    n_brightest: int | None = None,
    show_undistorted_catalog: bool = False,
) -> None:
    """Draw catalog, detection, and streak overlays onto an existing axes.

    Renders catalog-star crosshairs/circles, point-detection crosshairs, and
    boxes around streak detections and candidates, optionally including
    undistorted (SIP-removed) catalog positions.

    Args:
        ax: Matplotlib axes to draw onto.
        starfield: Solved starfield providing catalog positions and WCS.
        starlist: Optional star list whose centers are plotted instead.
        detections: Detected sources to overlay (points and streaks).
        streak: Fitted streak geometry used to draw line segments on catalog stars.
        streak_candidates: Candidate streaks to box (objects with ``.x``, ``.y``,
            ``.length_pixels``, ``.angle_deg``, ...).
        centercross: Draw catalog stars as center crosses rather than circles.
        marker: Marker style for catalog crosses; ``None`` disables them.
        markersize: Marker size / circle radius scale in points.
        linewidth: Line width for overlay strokes.
        n_brightest: If set, only overlay the N brightest catalog stars.
        show_undistorted_catalog: Also plot catalog positions computed without
            SIP distortion (as white squares) when the WCS carries SIP terms.
    """
    centers = None
    if starfield is not None:
        # Get all catalog stars
        catalog_stars = starfield.catalog_stars
        if catalog_stars and n_brightest is not None:
            # Filter to N brightest stars based on magnitude
            stars_with_mag = [
                (star, star.magnitude)
                for star in catalog_stars
                if star.magnitude is not None
            ]
            if stars_with_mag:
                # Sort by magnitude (brightest = lowest magnitude)
                stars_with_mag.sort(key=lambda x: x[1])
                # Take the N brightest
                brightest_stars = [star for star, _ in stars_with_mag[:n_brightest]]
                # Create a temporary starfield with only the brightest stars
                temp_starfield = StarField(
                    astrometric_fit_stars=starfield.astrometric_fit_stars,
                    catalog_stars=brightest_stars,
                    detections=starfield.detections,
                    image_metadata=starfield.image_metadata,
                    fit=starfield.fit,
                    wcs=starfield.wcs,
                    wcs_metadata=starfield.wcs_metadata,
                    detection_metadata=starfield.detection_metadata,
                    astrometry=starfield.astrometry,
                    wcs_status=starfield.wcs_status,
                    limiting_magnitude=starfield.limiting_magnitude,
                    fwhm_stats=starfield.fwhm_stats,
                    scale_factor=starfield.scale_factor,
                )
                centers = temp_starfield.catalog_centers_xy(
                    limiting_magnitude=starfield.limiting_magnitude
                )
            else:
                centers = starfield.catalog_centers_xy(
                    limiting_magnitude=starfield.limiting_magnitude
                )
        else:
            centers = starfield.catalog_centers_xy(
                limiting_magnitude=starfield.limiting_magnitude
            )

        # Plot undistorted catalog positions if requested and WCS has SIP
        if show_undistorted_catalog and starfield.wcs and catalog_stars:
            from astropy.wcs import WCS

            wcs_with_sip = starfield.wcs.to_astropy_wcs()
            header_with_sip = wcs_with_sip.to_header(relax=True)
            sip_order = header_with_sip.get("A_ORDER", 0)

            # Also check model directly
            if sip_order == 0 and starfield.wcs.A_ORDER:
                sip_order = starfield.wcs.A_ORDER
                logger.debug(f"Using SIP order from model: {sip_order}")

            # Also check WCS object directly
            if (
                sip_order == 0
                and hasattr(wcs_with_sip, "sip")
                and wcs_with_sip.sip is not None
            ) and hasattr(wcs_with_sip.sip, "a_order"):
                sip_order = wcs_with_sip.sip.a_order
                logger.debug(f"Using SIP order from WCS object: {sip_order}")

            if sip_order > 0:
                logger.debug(
                    f"Plotting undistorted catalog positions (SIP order={sip_order})"
                )
                # Create WCS without SIP
                header_no_sip = header_with_sip.copy()
                sip_keys_to_remove = []
                for key in list(header_no_sip.keys()):
                    if (
                        key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]
                        or key.startswith("A_")
                        or key.startswith("B_")
                        or key.startswith("AP_")
                        or key.startswith("BP_")
                    ):
                        sip_keys_to_remove.append(key)

                for key in sip_keys_to_remove:
                    del header_no_sip[key]

                # Also remove -SIP suffix from CTYPE if present
                if "CTYPE1" in header_no_sip and header_no_sip["CTYPE1"].endswith(
                    "-SIP"
                ):
                    header_no_sip["CTYPE1"] = header_no_sip["CTYPE1"][:-4]
                if "CTYPE2" in header_no_sip and header_no_sip["CTYPE2"].endswith(
                    "-SIP"
                ):
                    header_no_sip["CTYPE2"] = header_no_sip["CTYPE2"][:-4]

                try:
                    wcs_no_sip = WCS(header_no_sip, relax=True)

                    # Verify WCS without SIP works
                    test_ra, test_dec = wcs_no_sip.pixel_to_world_values(
                        centers[0][0] if len(centers) > 0 else 100,
                        centers[0][1] if len(centers) > 0 else 100,
                    )
                    logger.debug(
                        f"WCS without SIP test: pixel ({centers[0][0] if len(centers) > 0 else 100}, {centers[0][1] if len(centers) > 0 else 100}) -> RA={test_ra:.6f}, Dec={test_dec:.6f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create WCS without SIP for plotting: {e}")
                    wcs_no_sip = None

                if wcs_no_sip is not None:
                    # Convert catalog star RA/Dec to pixels using undistorted WCS
                    undistorted_centers = []
                    distorted_centers_sample = []

                    for star in catalog_stars:
                        if star.ra is not None and star.dec is not None:
                            try:
                                import astropy.units as u
                                from astropy.coordinates import SkyCoord

                                coords = SkyCoord(star.ra * u.deg, star.dec * u.deg)

                                # Get undistorted position (no SIP)
                                pix_no_sip = wcs_no_sip.world_to_pixel(coords)
                                x_undistorted = pix_no_sip[0] - 1  # Convert to 0-based
                                y_undistorted = pix_no_sip[1] - 1
                                undistorted_centers.append(
                                    [x_undistorted, y_undistorted]
                                )

                                # Also get distorted position (with SIP) for comparison
                                pix_with_sip = wcs_with_sip.world_to_pixel(coords)
                                x_distorted = pix_with_sip[0] - 1
                                y_distorted = pix_with_sip[1] - 1
                                distorted_centers_sample.append(
                                    [x_distorted, y_distorted]
                                )

                            except Exception as e:
                                logger.debug(
                                    f"Failed to convert star RA/Dec to pixels: {e}"
                                )
                                continue

                    if undistorted_centers:
                        undistorted_centers = np.array(undistorted_centers)
                        distorted_centers_sample = np.array(distorted_centers_sample)

                        # Log a sample comparison
                        if len(undistorted_centers) > 0:
                            sample_idx = len(undistorted_centers) // 2
                            dx = (
                                distorted_centers_sample[sample_idx][0]
                                - undistorted_centers[sample_idx][0]
                            )
                            dy = (
                                distorted_centers_sample[sample_idx][1]
                                - undistorted_centers[sample_idx][1]
                            )
                            radial_diff = np.sqrt(dx**2 + dy**2)
                            logger.debug(
                                f"Sample undistorted vs distorted difference: dx={dx:.2f}, dy={dy:.2f}, radial={radial_diff:.2f} px"
                            )

                        # Plot as white boxes (squares) - make them much larger
                        ax.scatter(
                            undistorted_centers[:, 0],
                            undistorted_centers[:, 1],
                            marker="s",  # square marker
                            s=markersize * 8,  # Much larger for visibility
                            facecolors="none",
                            edgecolors="white",
                            linewidths=max(2, linewidth * 2),  # Thicker lines
                            label="Catalog (undistorted, no SIP)",
                            alpha=0.9,
                            zorder=5,  # Plot on top
                        )
                    else:
                        logger.warning("No undistorted centers calculated")
                else:
                    logger.warning(
                        "Could not create WCS without SIP, skipping undistorted catalog plot"
                    )

    if starlist is not None:
        stars = starlist.centers_xy()
        centers = stars[:, :2] if len(stars.shape) > 1 else None

    if centers is not None and streak is not None:
        centercross = True
        marker = "+"
        # Calculate line segment endpoints for each center point
        half_length = streak.pixel_length / 2
        dx = half_length * streak.cosine_angle  # x offset from center
        dy = half_length * streak.sine_angle  # y offset from center

        # For each center point, create line segment from (x-dx,y-dy) to (x+dx,y+dy)
        for center in centers:
            x, y = center
            ax.plot(
                [x - dx, x + dx],
                [y - dy, y + dy],
                color="red",
                alpha=0.8,
                linewidth=linewidth,
            )

    if centers is not None and marker is not None and centers.shape[0] > 0:
        if centercross:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                marker=marker,
                color="red",
                s=markersize,
            )
        else:
            for center in centers:
                rect = patches.Circle(
                    center,
                    radius=2 * markersize,
                    linewidth=linewidth,
                    linestyle="-",
                    edgecolor="red",
                    facecolor="none",
                )

                ax.add_patch(rect)

    # Collect streak-type detection indices so we don't double-render them
    _streak_det_indices: set[int] = set()
    if detections and hasattr(detections, "detections"):
        for i, det in enumerate(detections.detections):
            if getattr(det, "detection_type", None) == "streak":
                _streak_det_indices.add(i)

    # Point-type detections: crosshairs (skip streak-type — they get a box below)
    if detections is not None:
        from matplotlib.patheffects import withStroke

        for i, center in enumerate(detections.centers_xy()):
            if i in _streak_det_indices:
                continue
            x, y = center[:2]
            fwhm = center[2]
            gap_size = fwhm * 4
            line_length = fwhm * 10

            ax.plot(
                [x - line_length, x - gap_size],
                [y, y],
                color="blue",
                linewidth=linewidth,
                path_effects=[withStroke(linewidth=linewidth * 3, foreground="white")],
            )
            ax.plot(
                [x, x],
                [y - line_length, y - gap_size],
                color="blue",
                linewidth=linewidth,
                path_effects=[withStroke(linewidth=linewidth * 3, foreground="white")],
            )

    # Streak-type detections + streak_candidates: prominent white box
    streak_items = list(streak_candidates or [])
    if detections and hasattr(detections, "detections"):
        for det in detections.detections:
            if getattr(det, "detection_type", None) == "streak" and det.angle_deg is not None:
                streak_items.append(det)

    for candidate in streak_items:
        cx, cy = candidate.x, candidate.y
        fwhm_w = getattr(candidate, "width_pixels", None) or getattr(candidate, "pixel_fwhm", None) or markersize
        length = getattr(candidate, "length_pixels", 0) or 0
        angle = getattr(candidate, "angle_deg", 0) or 0

        if length <= 0:
            continue

        # Box with padding so the thick line doesn't overlap the streak
        pad = fwhm_w * 2
        box_len = length + 2 * pad + fwhm_w
        box_wid = fwhm_w * 3 + 2 * pad
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        hl, hw = box_len / 2, box_wid / 2
        corners_x = [
            cx + hl * cos_a - hw * sin_a,
            cx + hl * cos_a + hw * sin_a,
            cx - hl * cos_a + hw * sin_a,
            cx - hl * cos_a - hw * sin_a,
            cx + hl * cos_a - hw * sin_a,
        ]
        corners_y = [
            cy + hl * sin_a + hw * cos_a,
            cy + hl * sin_a - hw * cos_a,
            cy - hl * sin_a - hw * cos_a,
            cy - hl * sin_a + hw * cos_a,
            cy + hl * sin_a + hw * cos_a,
        ]

        ax.plot(
            corners_x,
            corners_y,
            color="white",
            linewidth=linewidth * 1.5,
            alpha=0.9,
        )


def font_size(img: np.ndarray) -> float:
    """Compute a label font size scaled to the image's smaller dimension.

    Args:
        img: The image being annotated.

    Returns:
        A font size in points (at least 2).
    """
    return max(2, min(img.shape[1], img.shape[0]) * 0.01)


def plot_single_frame(
    img: np.ndarray,
    starfield: StarField | None = None,
    starlist: StarListImage | None = None,
    detections: StarListImage | None = None,
    streak: StreakMetadata | None = None,
    streak_candidates: list | None = None,
    output_file: str | Path | None = None,
    scale: bool = True,
    marker: str | None = "+",
    centercross: bool = False,
    markersize: float = 10,
    n_brightest: int | None = None,
    show_undistorted_catalog: bool = False,
    dpi: int | None = None,
    output_format: str | None = None,
    jpeg_quality: int = 95,
    png_compression: int = 6,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Render a single frame with an RA/Dec grid and detection overlays.

    When the starfield has a WCS fit, an equal-area RA/Dec grid with edge labels
    is drawn; otherwise a plain borderless image is produced. Overlays are added
    via :func:`plot_overs`. The result is either saved to ``output_file``
    (optionally re-encoded/optimized) or returned for further use.

    Args:
        img: Frame pixel data.
        starfield: Solved starfield providing catalog positions and WCS.
        starlist: Optional star list to overlay instead of the catalog.
        detections: Detected sources to overlay.
        streak: Fitted streak geometry for streak overlays.
        streak_candidates: Candidate streaks to box.
        output_file: Destination path; if ``None`` the figure/axes are returned.
        scale: Apply a zscale stretch before display.
        marker: Marker style for catalog crosses; ``None`` disables them.
        centercross: Draw catalog stars as center crosses rather than circles.
        markersize: Base marker size; overridden by the detection/streak FWHM
            when a fit is present.
        n_brightest: If set, only overlay the N brightest catalog stars.
        show_undistorted_catalog: Also plot SIP-removed catalog positions.
        dpi: Output DPI; defaults to 150, reduced to 75 for images over 4000 px.
        output_format: Output format (``"png"`` or ``"jpeg"``); inferred from the
            extension when ``None``.
        jpeg_quality: JPEG quality (1-100) when saving as JPEG.
        png_compression: PNG compression level (0-9) for the PIL optimization pass.

    Returns:
        The ``(figure, axes)`` pair when ``output_file`` is ``None``; otherwise
        ``None`` after saving.
    """
    logger.info(f"plotting frame {output_file if output_file else 'no output file'}")

    # Determine DPI - default to 150, but reduce for very large images to save space
    if dpi is None:
        # For images larger than 4kx4k, reduce DPI to save file size
        max_dimension = max(img.shape[0], img.shape[1])
        if max_dimension > 4000:
            dpi = 75  # Reduce DPI for very large images
            logger.info(
                f"Large image detected ({max_dimension}px), using reduced DPI={dpi} to save file size"
            )
        else:
            dpi = 150  # Default DPI for smaller images

    # Determine output format from file extension if not specified
    if output_file and output_format is None:
        output_path = Path(output_file)
        ext = output_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            output_format = "jpeg"
        elif ext == ".png":
            output_format = "png"
        # If no extension or unknown, default to PNG

    if starfield is not None and starfield.fit:
        # Get markersize from detection_metadata if available, otherwise use streak FWHM or default
        if (
            starfield.detection_metadata is not None
            and starfield.detection_metadata.pixel_fwhm is not None
        ):
            markersize = starfield.detection_metadata.pixel_fwhm
        elif streak is not None and streak.fwhm is not None:
            markersize = streak.fwhm
        else:
            markersize = 5.0  # Default fallback
        # Suppress FITSFixedWarning about WCS axes mismatch for wide-field images
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*WCS transformation has more axes.*"
            )
            wcs = starfield.wcs.to_astropy_wcs()

        # Get image dimensions
        height, width = img.shape

        fig = plt.figure(
            figsize=(1 * img.shape[1] / dpi, 1 * img.shape[0] / dpi),
            dpi=dpi,
            frameon=False,
        )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        ax = fig.add_subplot(111)
        ax.set_frame_on(False)

        # Sample a grid across the image to get all pixel coordinates
        # Use this to find grid lines that divide PIXELS (area) equally, not coordinates
        grid_density = 100  # Sample every N pixels
        y_grid, x_grid = np.mgrid[0:height:grid_density, 0:width:grid_density]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            coords = wcs.pixel_to_world(x_grid.flatten(), y_grid.flatten())
            ra_all = coords.ra.deg
            dec_all = coords.dec.deg

        # Filter valid
        valid = np.isfinite(ra_all) & np.isfinite(dec_all)
        ra_all = ra_all[valid]
        dec_all = dec_all[valid]

        # Handle RA wraparound using circular mean
        ra_rad = np.deg2rad(ra_all)
        ra_center = np.rad2deg(
            np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad)))
        )
        if ra_center < 0:
            ra_center += 360

        # Normalize all RAs around center
        ra_normalized = ra_all - ra_center
        ra_normalized = np.where(
            ra_normalized > 180, ra_normalized - 360, ra_normalized
        )
        ra_normalized = np.where(
            ra_normalized < -180, ra_normalized + 360, ra_normalized
        )

        # Find percentiles to divide pixels equally (3 lines = 4 bins = 25%, 50%, 75%)
        ra_percentiles = np.percentile(ra_normalized, [25, 50, 75])
        dec_percentiles = np.percentile(dec_all, [25, 50, 75])

        # Convert back to standard RA range
        ra_ticks_normalized = ra_percentiles
        ra_ticks = ra_ticks_normalized + ra_center
        ra_ticks = np.where(ra_ticks < 0, ra_ticks + 360, ra_ticks)
        ra_ticks = np.where(ra_ticks >= 360, ra_ticks - 360, ra_ticks)

        dec_ticks = dec_percentiles

        logger.debug(f"RA grid lines (equal area): {ra_ticks}")
        logger.debug(f"Dec grid lines (equal area): {dec_ticks}")

        # Don't use linspace, use the percentile-based ticks directly

        # Grid lines already calculated above using percentiles
        # (ra_ticks and dec_ticks are set based on equal pixel area division)

        # Function to place labels
        def place_label(
            x: float,
            y: float,
            label_text: str,
            angle: float,
            on_x_axis: bool = True,
        ) -> None:
            """Place a rotated white label at ``(x, y)`` if inside the image.

            Args:
                x: Label x-position in pixels.
                y: Label y-position in pixels.
                label_text: Text to render.
                angle: Text rotation in degrees.
                on_x_axis: Whether the label sits on the x-axis (affects padding
                    and alignment).
            """
            if x >= 0 and x < width and y >= 0 and y < height:
                fs = font_size(img)
                if on_x_axis:
                    hpad = 0
                    vpad = 0
                    ha = "left"
                    va = "bottom"
                else:
                    vpad = 0.2 * fs
                    hpad = 0
                    ha = "left"
                    va = "bottom" if angle >= 0 else "top"

                ax.text(
                    x + hpad,
                    y + vpad,
                    label_text,
                    color="white",
                    ha=ha,
                    va=va,
                    size=fs,
                    rotation=angle,
                )

        # Helper function to normalize angle to [-90, 90] degrees
        def normalize_angle(angle: float) -> float:
            """Fold an angle into the ``[-90, 90]`` degree range.

            Args:
                angle: Angle in degrees.

            Returns:
                The equivalent angle within ``[-90, 90]`` degrees.
            """
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            return angle

        # Draw RA grid lines (sample many points for curved lines)
        n_samples = (
            200  # Increased samples for smoother curves and better boundary detection
        )
        # Use the actual Dec range from our sampled data, but extend it significantly to ensure we cross boundaries
        dec_sampling_min = np.min(dec_all)
        dec_sampling_max = np.max(dec_all)
        # Extend the range by 100% (double it) to ensure we cross image boundaries
        # This is needed because curved grid lines may not reach edges with small extensions
        dec_range = dec_sampling_max - dec_sampling_min
        if dec_range > 0:
            dec_sampling_min_extended = (
                dec_sampling_min - dec_range
            )  # Extend by full range
            dec_sampling_max_extended = dec_sampling_max + dec_range
        else:
            # If range is zero or very small, use a fixed extension
            dec_sampling_min_extended = dec_sampling_min - 1.0
            dec_sampling_max_extended = dec_sampling_max + 1.0

        for ra in ra_ticks:
            try:
                dec_samples = np.linspace(
                    dec_sampling_min_extended, dec_sampling_max_extended, n_samples
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*failed to converge.*")
                    warnings.filterwarnings(
                        "ignore", message=".*All-NaN slice encountered.*"
                    )
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    coords = SkyCoord(
                        ra=np.full(n_samples, ra) * u.deg, dec=dec_samples * u.deg
                    )
                    x_coords, y_coords = wcs.world_to_pixel(coords)

                # Only filter out non-finite coordinates, let matplotlib clip to bounds
                valid = np.isfinite(x_coords) & np.isfinite(y_coords)

                if not np.any(valid):
                    continue

                x_coords = x_coords[valid]
                y_coords = y_coords[valid]

                if len(x_coords) < 2:
                    continue

            except Exception as e:
                logger.debug(f"Skipping RA grid line at {ra}° due to error: {e}")
                continue

            # Draw the curved line - brighter alpha
            ax.plot(
                x_coords,
                y_coords,
                color="white",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )

            ra_text = f"{SkyCoord(ra=ra * u.deg, dec=0 * u.deg).ra.to_string(unit=u.hour, sep=':', pad=True, format='hms', precision=1)}s".replace(
                ":", "h", 1
            ).replace(
                ":", "m", 1
            )

            # Find visible points
            visible = (
                (x_coords >= 0)
                & (x_coords < width)
                & (y_coords >= 0)
                & (y_coords < height)
            )
            if np.sum(visible) < 2:
                logger.debug(
                    f"Skipping RA {ra:.2f}° label: only {np.sum(visible)} visible points"
                )
                continue

            # For RA lines: check all 4 edges to find where this RA value appears
            # Sample all edges and find where target RA is closest
            best_pos = None
            best_diff = float("inf")
            best_edge = None

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    # Check bottom edge (y = height - 1)
                    bottom_x = np.arange(0, width)
                    bottom_pixels = np.column_stack(
                        [bottom_x, np.full(width, height - 1)]
                    )
                    bottom_world = wcs.pixel_to_world(
                        bottom_pixels[:, 0], bottom_pixels[:, 1]
                    )
                    bottom_ra = bottom_world.ra.deg
                    ra_diffs = bottom_ra - ra
                    ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
                    ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)
                    min_idx = np.argmin(np.abs(ra_diffs))
                    if np.abs(ra_diffs[min_idx]) < best_diff:
                        best_diff = np.abs(ra_diffs[min_idx])
                        best_pos = (bottom_x[min_idx], height - 1)
                        best_edge = "bottom"

                    # Check top edge (y = 0)
                    top_x = np.arange(0, width)
                    top_pixels = np.column_stack([top_x, np.zeros(width)])
                    top_world = wcs.pixel_to_world(top_pixels[:, 0], top_pixels[:, 1])
                    top_ra = top_world.ra.deg
                    ra_diffs = top_ra - ra
                    ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
                    ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)
                    min_idx = np.argmin(np.abs(ra_diffs))
                    if np.abs(ra_diffs[min_idx]) < best_diff:
                        best_diff = np.abs(ra_diffs[min_idx])
                        best_pos = (top_x[min_idx], 0)
                        best_edge = "top"

                    # Check left edge (x = 0)
                    left_y = np.arange(0, height)
                    left_pixels = np.column_stack([np.zeros(height), left_y])
                    left_world = wcs.pixel_to_world(
                        left_pixels[:, 0], left_pixels[:, 1]
                    )
                    left_ra = left_world.ra.deg
                    ra_diffs = left_ra - ra
                    ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
                    ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)
                    min_idx = np.argmin(np.abs(ra_diffs))
                    if np.abs(ra_diffs[min_idx]) < best_diff:
                        best_diff = np.abs(ra_diffs[min_idx])
                        best_pos = (0, left_y[min_idx])
                        best_edge = "left"

                    # Check right edge (x = width - 1)
                    right_y = np.arange(0, height)
                    right_pixels = np.column_stack(
                        [np.full(height, width - 1), right_y]
                    )
                    right_world = wcs.pixel_to_world(
                        right_pixels[:, 0], right_pixels[:, 1]
                    )
                    right_ra = right_world.ra.deg
                    ra_diffs = right_ra - ra
                    ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
                    ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)
                    min_idx = np.argmin(np.abs(ra_diffs))
                    if np.abs(ra_diffs[min_idx]) < best_diff:
                        best_diff = np.abs(ra_diffs[min_idx])
                        best_pos = (width - 1, right_y[min_idx])
                        best_edge = "right"

                if best_pos is not None:
                    x_at_bottom = best_pos[0]
                    bottom_edge_y = best_pos[
                        1
                    ]  # Actually the y position on whichever edge
                else:
                    x_at_bottom = width / 2
                    bottom_edge_y = height - 1

            except Exception as e:
                logger.debug(f"Failed to find RA intersection: {e}, using center")
                x_at_bottom = width / 2
                bottom_edge_y = height - 1

            # Calculate angle at the label position
            # Use points near the intersection to calculate the grid line direction
            if len(x_coords) >= 2:
                # Find the point closest to the intersection position
                dists = np.sqrt(
                    (x_coords - x_at_bottom) ** 2 + (y_coords - bottom_edge_y) ** 2
                )
                closest_idx = np.argmin(dists)
                if closest_idx > 0 and closest_idx < len(x_coords) - 1:
                    # Use neighbors to get direction
                    dy = y_coords[closest_idx + 1] - y_coords[closest_idx - 1]
                    dx = x_coords[closest_idx + 1] - x_coords[closest_idx - 1]
                elif closest_idx > 0:
                    dy = y_coords[closest_idx] - y_coords[closest_idx - 1]
                    dx = x_coords[closest_idx] - x_coords[closest_idx - 1]
                elif closest_idx < len(x_coords) - 1:
                    dy = y_coords[closest_idx + 1] - y_coords[closest_idx]
                    dx = x_coords[closest_idx + 1] - x_coords[closest_idx]
                else:
                    # Fallback: use overall direction of the line
                    dy = y_coords[-1] - y_coords[0]
                    dx = x_coords[-1] - x_coords[0]
            else:
                dy = 0
                dx = 1  # Default horizontal

            angle = -np.degrees(np.arctan2(dy, dx))

            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # Place label - only on left or bottom edges
            # If found on top/right, project to bottom/left
            margin = int(0.005 * width)  # 0.5% of image width

            if best_edge == "bottom":
                label_x = x_at_bottom
                label_y = bottom_edge_y - margin
                ha = "center"
                va = "bottom"
            elif best_edge == "top":
                # Project to bottom edge
                bottom_x = np.arange(0, width)
                bottom_pixels = np.column_stack([bottom_x, np.full(width, height - 1)])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    bottom_world = wcs.pixel_to_world(
                        bottom_pixels[:, 0], bottom_pixels[:, 1]
                    )
                    bottom_ra = bottom_world.ra.deg
                    ra_diffs = bottom_ra - ra
                    ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
                    ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)
                    min_idx = np.argmin(np.abs(ra_diffs))
                    label_x = bottom_x[min_idx]
                label_y = height - 1 - margin
                ha = "center"
                va = "bottom"
            elif best_edge == "left":
                label_x = margin
                label_y = bottom_edge_y
                ha = "left"
                va = "center"
            else:  # right
                # Project to left edge
                left_y = np.arange(0, height)
                left_pixels = np.column_stack([np.zeros(height), left_y])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    left_world = wcs.pixel_to_world(
                        left_pixels[:, 0], left_pixels[:, 1]
                    )
                    left_ra = left_world.ra.deg
                    ra_diffs = left_ra - ra
                    ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
                    ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)
                    min_idx = np.argmin(np.abs(ra_diffs))
                    label_y = left_y[min_idx]
                label_x = margin
                ha = "left"
                va = "center"

            ax.text(
                label_x,
                label_y,
                ra_text,
                color="white",
                ha=ha,
                va=va,
                size=font_size(img),
                rotation=angle,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "black",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )
            logger.debug(
                f"RA {ra:.2f}°: pos=({label_x:.0f},{label_y:.0f}), edge={best_edge}, angle={angle:.1f}°"
            )

        # Draw Dec grid lines
        # Use the actual RA range from our sampled data (in normalized space)
        ra_sampling_min = np.min(ra_normalized) + ra_center
        ra_sampling_max = np.max(ra_normalized) + ra_center
        # Normalize to [0, 360)
        if ra_sampling_min < 0:
            ra_sampling_min += 360
        if ra_sampling_max < 0:
            ra_sampling_max += 360

        # Extend the RA range to ensure we cross image boundaries
        # Extend by 100% (double it) to ensure we cross image boundaries
        if ra_sampling_max < ra_sampling_min:
            # Crosses 0°, extend in normalized space
            ra_range_normalized = (ra_sampling_max - ra_sampling_min + 360) % 360
            if ra_range_normalized == 0:
                ra_range_normalized = 360
            ra_sampling_min_extended = (
                ra_sampling_min - ra_range_normalized
            ) % 360  # Extend by full range
            ra_sampling_max_extended = (ra_sampling_max + ra_range_normalized) % 360
        else:
            ra_range = ra_sampling_max - ra_sampling_min
            if ra_range > 0:
                ra_sampling_min_extended = (
                    ra_sampling_min - ra_range
                ) % 360  # Extend by full range
                ra_sampling_max_extended = (ra_sampling_max + ra_range) % 360
            else:
                # If range is zero or very small, use a fixed extension
                ra_sampling_min_extended = (ra_sampling_min - 5.0) % 360
                ra_sampling_max_extended = (ra_sampling_max + 5.0) % 360

        for dec in dec_ticks:
            try:
                # Need to handle RA wraparound for sampling
                if ra_sampling_max_extended < ra_sampling_min_extended:
                    # Crosses 0°, sample in normalized space then convert
                    ra_samples_normalized = np.linspace(
                        ra_sampling_min_extended - ra_center,
                        ra_sampling_max_extended - ra_center + 360,
                        n_samples,
                    )
                    ra_samples = (ra_samples_normalized + ra_center) % 360
                else:
                    ra_samples = np.linspace(
                        ra_sampling_min_extended, ra_sampling_max_extended, n_samples
                    )

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*failed to converge.*")
                    warnings.filterwarnings(
                        "ignore", message=".*All-NaN slice encountered.*"
                    )
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    coords = SkyCoord(
                        ra=ra_samples * u.deg, dec=np.full(n_samples, dec) * u.deg
                    )
                    x_coords, y_coords = wcs.world_to_pixel(coords)

                # Only filter out non-finite coordinates, let matplotlib clip to bounds
                valid = np.isfinite(x_coords) & np.isfinite(y_coords)

                if not np.any(valid):
                    continue

                x_coords = x_coords[valid]
                y_coords = y_coords[valid]

                if len(x_coords) < 2:
                    continue

            except Exception as e:
                logger.debug(f"Skipping Dec grid line at {dec}° due to error: {e}")
                continue

            # Draw the curved line - brighter alpha
            ax.plot(
                x_coords,
                y_coords,
                color="white",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )

            dec_text = f"{SkyCoord(ra=0 * u.deg, dec=dec * u.deg).dec.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True)}''".replace(
                ":", "°", 1
            ).replace(
                ":", "'", 1
            )

            # Find visible points
            visible = (
                (x_coords >= 0)
                & (x_coords < width)
                & (y_coords >= 0)
                & (y_coords < height)
            )
            if np.sum(visible) < 2:
                logger.debug(
                    f"Skipping Dec {dec:.2f}° label: only {np.sum(visible)} visible points"
                )
                continue

            # For Dec lines: check all 4 edges to find where this Dec value appears
            # Sample all edges and find where target Dec is closest
            best_pos = None
            best_diff = float("inf")
            best_edge = None

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    # Check bottom edge (y = height - 1)
                    bottom_x = np.arange(0, width)
                    bottom_pixels = np.column_stack(
                        [bottom_x, np.full(width, height - 1)]
                    )
                    bottom_world = wcs.pixel_to_world(
                        bottom_pixels[:, 0], bottom_pixels[:, 1]
                    )
                    bottom_dec = bottom_world.dec.deg
                    dec_diffs = np.abs(bottom_dec - dec)
                    min_idx = np.argmin(dec_diffs)
                    if dec_diffs[min_idx] < best_diff:
                        best_diff = dec_diffs[min_idx]
                        best_pos = (bottom_x[min_idx], height - 1)
                        best_edge = "bottom"

                    # Check top edge (y = 0)
                    top_x = np.arange(0, width)
                    top_pixels = np.column_stack([top_x, np.zeros(width)])
                    top_world = wcs.pixel_to_world(top_pixels[:, 0], top_pixels[:, 1])
                    top_dec = top_world.dec.deg
                    dec_diffs = np.abs(top_dec - dec)
                    min_idx = np.argmin(dec_diffs)
                    if dec_diffs[min_idx] < best_diff:
                        best_diff = dec_diffs[min_idx]
                        best_pos = (top_x[min_idx], 0)
                        best_edge = "top"

                    # Check left edge (x = 0)
                    left_y = np.arange(0, height)
                    left_pixels = np.column_stack([np.zeros(height), left_y])
                    left_world = wcs.pixel_to_world(
                        left_pixels[:, 0], left_pixels[:, 1]
                    )
                    left_dec = left_world.dec.deg
                    dec_diffs = np.abs(left_dec - dec)
                    min_idx = np.argmin(dec_diffs)
                    if dec_diffs[min_idx] < best_diff:
                        best_diff = dec_diffs[min_idx]
                        best_pos = (0, left_y[min_idx])
                        best_edge = "left"

                    # Check right edge (x = width - 1)
                    right_y = np.arange(0, height)
                    right_pixels = np.column_stack(
                        [np.full(height, width - 1), right_y]
                    )
                    right_world = wcs.pixel_to_world(
                        right_pixels[:, 0], right_pixels[:, 1]
                    )
                    right_dec = right_world.dec.deg
                    dec_diffs = np.abs(right_dec - dec)
                    min_idx = np.argmin(dec_diffs)
                    if dec_diffs[min_idx] < best_diff:
                        best_diff = dec_diffs[min_idx]
                        best_pos = (width - 1, right_y[min_idx])
                        best_edge = "right"

                if best_pos is not None:
                    left_edge_x = best_pos[
                        0
                    ]  # Actually the x position on whichever edge
                    y_at_left = best_pos[1]
                else:
                    left_edge_x = 0
                    y_at_left = height / 2

            except Exception as e:
                logger.debug(f"Failed to find Dec intersection: {e}, using center")
                left_edge_x = 0
                y_at_left = height / 2
            # Calculate angle at the label position
            # Use points near the intersection to calculate the grid line direction
            if len(x_coords) >= 2:
                # Find the point closest to the intersection position
                dists = np.sqrt(
                    (x_coords - left_edge_x) ** 2 + (y_coords - y_at_left) ** 2
                )
                closest_idx = np.argmin(dists)
                if closest_idx > 0 and closest_idx < len(x_coords) - 1:
                    # Use neighbors to get direction
                    dy = y_coords[closest_idx + 1] - y_coords[closest_idx - 1]
                    dx = x_coords[closest_idx + 1] - x_coords[closest_idx - 1]
                elif closest_idx > 0:
                    dy = y_coords[closest_idx] - y_coords[closest_idx - 1]
                    dx = x_coords[closest_idx] - x_coords[closest_idx - 1]
                elif closest_idx < len(x_coords) - 1:
                    dy = y_coords[closest_idx + 1] - y_coords[closest_idx]
                    dx = x_coords[closest_idx + 1] - x_coords[closest_idx]
                else:
                    # Fallback: use overall direction of the line
                    dy = y_coords[-1] - y_coords[0]
                    dx = x_coords[-1] - x_coords[0]
            else:
                dy = 1
                dx = 0  # Default vertical

            angle = -np.degrees(np.arctan2(dy, dx))

            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # Place label - only on left or bottom edges
            # If found on top/right, project to bottom/left
            margin = int(0.005 * width)  # 0.5% of image width

            if best_edge == "left":
                label_x = margin
                label_y = y_at_left
                ha = "left"
                va = "center"
            elif best_edge == "right":
                # Project to left edge
                left_y = np.arange(0, height)
                left_pixels = np.column_stack([np.zeros(height), left_y])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    left_world = wcs.pixel_to_world(
                        left_pixels[:, 0], left_pixels[:, 1]
                    )
                    left_dec = left_world.dec.deg
                    dec_diffs = np.abs(left_dec - dec)
                    min_idx = np.argmin(dec_diffs)
                    label_y = left_y[min_idx]
                label_x = margin
                ha = "left"
                va = "center"
            elif best_edge == "bottom":
                label_x = left_edge_x
                label_y = y_at_left - margin
                ha = "center"
                va = "bottom"
            else:  # top
                # Project to bottom edge
                bottom_x = np.arange(0, width)
                bottom_pixels = np.column_stack([bottom_x, np.full(width, height - 1)])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    bottom_world = wcs.pixel_to_world(
                        bottom_pixels[:, 0], bottom_pixels[:, 1]
                    )
                    bottom_dec = bottom_world.dec.deg
                    dec_diffs = np.abs(bottom_dec - dec)
                    min_idx = np.argmin(dec_diffs)
                    label_x = bottom_x[min_idx]
                label_y = height - 1 - margin
                ha = "center"
                va = "bottom"

            ax.text(
                label_x,
                label_y,
                dec_text,
                color="white",
                ha=ha,
                va=va,
                size=font_size(img),
                rotation=angle,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "black",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )
            logger.debug(
                f"Dec {dec:.2f}°: pos=({label_x:.0f},{label_y:.0f}), edge={best_edge}, angle={angle:.1f}°"
            )
    else:
        wcs = None
        fig, ax = prep_axes(*img.shape)

    if scale:
        ax.imshow(zscale(img), cmap="gray")
    else:
        ax.imshow(img, cmap="gray")

    plot_overs(
        ax,
        starfield=starfield,
        starlist=starlist,
        detections=detections,
        streak=streak,
        streak_candidates=streak_candidates,
        centercross=centercross,
        marker=marker,
        markersize=markersize,
        n_brightest=n_brightest,
        show_undistorted_catalog=show_undistorted_catalog,
    )

    ax.set_xlim(0, img.shape[1] - 1)
    ax.set_ylim(img.shape[0] - 1, 0)

    shift = 0.005 * np.array(img.shape)
    # here I'd like to add in white text at (width, height), "SENPAI v0.1"
    ax.text(
        img.shape[1] - shift[1],
        shift[0],
        f"SENPAI v{senpai.__version__}",
        color="white",
        ha="right",
        va="top",
        size=font_size(img) * 1.5,
    )

    logger.debug("plotting complete")
    if output_file:
        # Prepare savefig kwargs based on format
        save_kwargs = {"dpi": dpi}
        if output_format == "jpeg" or output_format == "jpg":
            save_kwargs["format"] = "jpeg"
            save_kwargs["quality"] = jpeg_quality
            save_kwargs["optimize"] = True
            logger.info(f"Saving as JPEG with quality={jpeg_quality}, DPI={dpi}")
        elif output_format == "png":
            save_kwargs["format"] = "png"
            logger.info(f"Saving as PNG with DPI={dpi}")
        else:
            # Default PNG behavior
            logger.info(f"Saving with DPI={dpi}")

        plt.savefig(output_file, **save_kwargs)

        # For PNG files, try to optimize further using PIL if available
        output_path = Path(output_file)
        if (output_format is None or output_format == "png") and output_path.suffix.lower() == ".png":
            try:
                from PIL import Image

                img_pil = Image.open(output_file)
                # Convert to RGB if needed (removes alpha channel if present, saves space)
                if img_pil.mode in ("RGBA", "LA", "P"):
                    # Convert to RGB
                    rgb_img = Image.new("RGB", img_pil.size, (255, 255, 255))
                    if img_pil.mode == "P":
                        img_pil = img_pil.convert("RGBA")
                    rgb_img.paste(
                        img_pil,
                        mask=(
                            img_pil.split()[-1]
                            if img_pil.mode in ("RGBA", "LA")
                            else None
                        ),
                    )
                    img_pil = rgb_img
                # Save with optimization
                img_pil.save(
                    output_file,
                    "PNG",
                    optimize=True,
                    compress_level=png_compression,
                )
                logger.debug(
                    f"Optimized PNG file size with compression level={png_compression}"
                )
            except ImportError:
                pass  # PIL not available, skip optimization
            except Exception as e:
                logger.debug(f"Could not optimize PNG: {e}")

        plt.close()

        # Log file size for debugging
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved plot to {output_file} ({file_size_mb:.1f} MB)")
    else:
        logger.info("returning figure and axes")
        return fig, ax


def plot_photometry_frame(
    img: np.ndarray,
    apertures: Aperture | None = None,
    annuli: Aperture | None = None,
    output_file: str | Path | None = None,
    scale: bool = True,
) -> None:
    """Render a frame with photometric aperture and background-annulus overlays.

    Args:
        img: Frame pixel data (typically the per-pixel counts array).
        apertures: Photutils source apertures to draw in white.
        annuli: Photutils background annuli to draw in red.
        output_file: Destination path; if ``None`` the figure is shown instead.
        scale: Apply a zscale stretch before display.
    """
    _fig, ax = prep_axes(*img.shape)

    # minval = np.min(img.flatten())
    # maxval = (np.median(img.flatten()) - minval) * 4 + minval
    if scale:
        ax.imshow(zscale(img), cmap="gray")
    else:
        ax.imshow(img, cmap="gray")

    apertures.plot(color=(1, 1, 1, 1))
    annuli.plot(color=(1, 0, 0, 1))

    if output_file:
        plt.savefig(output_file)

    else:
        plt.show()

    plt.close()


def plot_sip_distortions(
    wcs: WCS,
    grid_spacing: int = 50,
    plot_type: str = "arrows",
    output_file: str | None = None,
    figsize: tuple = (10, 8),
    **kwargs: float,
) -> tuple | None:
    """Plot SIP (Simple Imaging Polynomial) distortions across the field of view.

    Args:
        wcs: Astropy WCS object containing SIP coefficients.
        grid_spacing: Spacing between grid points in pixels.
        plot_type: Type of plot - "arrows" for a quiver plot, "contours" for
            distortion-magnitude contours, "separate" for individual dx/dy/
            magnitude panels.
        output_file: Optional file path to save the plot.
        figsize: Figure size for the plot.
        **kwargs: Additional numeric arguments passed to the plotting functions
            (e.g. ``arrow_scale``, ``quiver_scale``, ``arrow_width``).

    Returns:
        The ``(fig, ax)`` pair if ``output_file`` is ``None``, otherwise ``None``.

    Raises:
        ValueError: If ``plot_type`` is not one of the recognized values.
    """
    # Check if WCS has SIP distortions
    has_sip = False
    if (hasattr(wcs, "sip") and wcs.sip is not None) or wcs.sip is not None:
        has_sip = True

    if not has_sip:
        logger.warning("No SIP distortions found in WCS")
        return None

    # Get image dimensions from WCS
    if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
        height, width = wcs.pixel_shape
    else:
        # Try to get from WCS header NAXIS
        header = wcs.to_header()
        if "NAXIS1" in header and "NAXIS2" in header:
            width = int(header["NAXIS1"])
            height = int(header["NAXIS2"])
        else:
            # Default fallback
            height, width = 1024, 1024
            logger.warning(
                f"Could not determine image shape from WCS, using default {height}x{width}"
            )

    # Create grid of pixel coordinates
    y_coords, x_coords = np.mgrid[0:height:grid_spacing, 0:width:grid_spacing]

    # Flatten coordinate arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Convert pixel coordinates to world coordinates using full WCS (with SIP)
    world_coords_with_sip = wcs.pixel_to_world(x_flat, y_flat)

    # Create WCS without SIP for comparison
    header_no_sip = wcs.to_header().copy()

    # Remove SIP keywords
    sip_keys_to_remove = []
    for key in list(header_no_sip.keys()):
        if (
            key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]
            or key.startswith("A_")
            or key.startswith("B_")
            or key.startswith("AP_")
            or key.startswith("BP_")
        ):
            sip_keys_to_remove.append(key)

    for key in sip_keys_to_remove:
        del header_no_sip[key]

    # Remove -SIP suffix from CTYPE if present
    if "CTYPE1" in header_no_sip and header_no_sip["CTYPE1"].endswith("-SIP"):
        header_no_sip["CTYPE1"] = header_no_sip["CTYPE1"][:-4]
    if "CTYPE2" in header_no_sip and header_no_sip["CTYPE2"].endswith("-SIP"):
        header_no_sip["CTYPE2"] = header_no_sip["CTYPE2"][:-4]

    try:
        wcs_no_sip = WCS(header_no_sip, relax=True)
    except Exception as e:
        logger.error(f"Failed to create WCS without SIP: {e}")
        return None

    # Convert world coordinates back to pixel coordinates using linear WCS (no SIP)
    pixel_coords_no_sip = wcs_no_sip.world_to_pixel(world_coords_with_sip)

    # Calculate distortion vectors (difference between SIP and linear positions)
    dx = x_flat - pixel_coords_no_sip[0]  # SIP pixel - linear pixel
    dy = y_flat - pixel_coords_no_sip[1]
    distortion_magnitude = np.sqrt(dx**2 + dy**2)

    # Create the plot
    if plot_type == "separate":
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("SIP Distortions Across Field of View")

        # Plot dx distortions
        im1 = axes[0].scatter(y_flat, x_flat, c=dx, cmap="RdBu_r", s=20, alpha=0.8)
        axes[0].set_title("X Distortion (pixels)")
        axes[0].set_xlabel("Y pixel")
        axes[0].set_ylabel("X pixel")
        axes[0].set_aspect("equal")
        axes[0].set_xlim(0, height - 1)
        axes[0].set_ylim(width - 1, 0)
        plt.colorbar(im1, ax=axes[0], label="dx (pixels)")

        # Plot dy distortions
        im2 = axes[1].scatter(y_flat, x_flat, c=dy, cmap="RdBu_r", s=20, alpha=0.8)
        axes[1].set_title("Y Distortion (pixels)")
        axes[1].set_xlabel("Y pixel")
        axes[1].set_ylabel("X pixel")
        axes[1].set_aspect("equal")
        axes[1].set_xlim(0, height - 1)
        axes[1].set_ylim(width - 1, 0)
        plt.colorbar(im2, ax=axes[1], label="dy (pixels)")

        # Plot distortion magnitude
        im3 = axes[2].scatter(
            y_flat, x_flat, c=distortion_magnitude, cmap="viridis", s=20, alpha=0.8
        )
        axes[2].set_title("Distortion Magnitude (pixels)")
        axes[2].set_xlabel("Y pixel")
        axes[2].set_ylabel("X pixel")
        axes[2].set_aspect("equal")
        axes[2].set_xlim(0, height - 1)
        axes[2].set_ylim(width - 1, 0)
        plt.colorbar(im3, ax=axes[2], label="Magnitude (pixels)")

    elif plot_type == "contours":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle("SIP Distortion Magnitude Contours")

        # Create contour plot of distortion magnitude
        levels = np.linspace(0, np.max(distortion_magnitude), 20)
        cs = ax.tricontourf(
            y_flat,
            x_flat,
            distortion_magnitude,
            levels=levels,
            cmap="viridis",
            alpha=0.8,
        )
        ax.tricontour(
            y_flat,
            x_flat,
            distortion_magnitude,
            levels=levels,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )

        ax.set_title("SIP Distortion Magnitude (pixels)")
        ax.set_xlabel("Y pixel")
        ax.set_ylabel("X pixel")
        ax.set_aspect("equal")
        ax.set_xlim(0, height - 1)
        ax.set_ylim(width - 1, 0)
        plt.colorbar(cs, ax=ax, label="Distortion (pixels)")

    elif plot_type == "arrows":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle("SIP Distortions Across Field of View")

        # Create quiver plot showing distortion vectors
        # Filter out small distortions for clarity
        significant = distortion_magnitude > np.percentile(
            distortion_magnitude, 10
        )  # Top 90%

        if np.any(significant):
            # Scale arrow lengths for visibility
            scale_factor = kwargs.get("arrow_scale", 1.0)
            dx_scaled = dx[significant] * scale_factor
            dy_scaled = dy[significant] * scale_factor

            # Plot arrows
            q = ax.quiver(
                y_flat[significant],
                x_flat[significant],
                dy_scaled,
                dx_scaled,
                distortion_magnitude[significant],
                cmap="viridis",
                scale=kwargs.get("quiver_scale", 1),
                scale_units="xy",
                angles="xy",
                alpha=0.8,
                width=kwargs.get("arrow_width", 0.005),
                headwidth=kwargs.get("arrow_headwidth", 3),
                headlength=kwargs.get("arrow_headlength", 5),
            )

            # Add colorbar
            plt.colorbar(q, ax=ax, label="Distortion magnitude (pixels)")

            # Add grid points without arrows (small distortions)
            ax.scatter(
                y_flat[~significant],
                x_flat[~significant],
                c="gray",
                s=5,
                alpha=0.3,
                label="Small distortions",
            )

        ax.set_title("SIP Distortion Vectors")
        ax.set_xlabel("Y pixel")
        ax.set_ylabel("X pixel")
        ax.set_aspect("equal")
        ax.set_xlim(0, height - 1)
        ax.set_ylim(width - 1, 0)
        ax.legend()

    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Must be 'arrows', 'contours', or 'separate'"
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    else:
        return fig, plt.gca()


def plot_shift_validation(
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    valid_stars: list,
    source_fluxes: np.ndarray,
    target_fluxes: np.ndarray,
    shift_x: float,
    shift_y: float,
    correlation: float,
    median_ratio: float,
    ratio_std: float,
    box_size: int,
    half_box: int,
    source_index: int,
    target_index: int,
) -> None:
    """Render diagnostic plots comparing a source frame to a shifted target frame.

    Saves comparison, close-up, and flux-correlation figures to the configured
    output directory to validate an inter-frame pixel shift.

    Args:
        source_frame (np.ndarray): Source frame image data.
        target_frame (np.ndarray): Target frame image data.
        valid_stars (list): Star tuples ``(x, y, x_shifted, y_shifted)`` matched
            between the two frames.
        source_fluxes (np.ndarray): Measured fluxes of stars in the source frame.
        target_fluxes (np.ndarray): Measured fluxes of stars in the target frame.
        shift_x (float): Applied shift in x (pixels).
        shift_y (float): Applied shift in y (pixels).
        correlation (float): Flux correlation between source and target.
        median_ratio (float): Median source-to-target flux ratio.
        ratio_std (float): Standard deviation of the flux ratio.
        box_size (int): Side length of the star bounding boxes (pixels).
        half_box (int): Half the box size (pixels).
        source_index (int): Index of the source frame.
        target_index (int): Index of the target frame.

    Returns:
        None
    """

    def simple_scale(image: np.ndarray) -> np.ndarray:
        """Linearly stretch an image to [0, 1] using its 1st/99th percentiles."""
        vmin, vmax = np.percentile(image, [1, 99])
        return np.clip((image - vmin) / (vmax - vmin), 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(zscale(source_frame), cmap="viridis", origin="upper")
    ax1.set_title("Source Frame")
    ax2.imshow(zscale(target_frame), cmap="viridis", origin="upper")
    ax2.set_title(f"Target Frame (Shift: {shift_x:.1f}, {shift_y:.1f})")

    for i, (x, y, x_shifted, y_shifted) in enumerate(valid_stars[:15]):
        rect1 = patches.Rectangle(
            (x - half_box, y - half_box),
            box_size,
            box_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax1.add_patch(rect1)
        ax1.text(
            x,
            y - half_box - 5,
            f"{i}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.5},
        )

        rect2 = patches.Rectangle(
            (x_shifted - half_box, y_shifted - half_box),
            box_size,
            box_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax2.add_patch(rect2)
        ax2.text(
            x_shifted,
            y_shifted - half_box - 5,
            f"{i}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.5},
        )

        ax1.plot(x, y, "g+", markersize=6)
        ax2.plot(x_shifted, y_shifted, "g+", markersize=6)

    ax1.plot([], [], "r-", linewidth=1, label="Star box")
    ax1.plot([], [], "g+", markersize=6, label="Star center")
    ax1.legend(loc="upper right")

    plt.figtext(
        0.5,
        0.95,
        f"Shift: ({shift_x:.1f}, {shift_y:.1f}) pixels",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(
        Path(settings.plotting.output_dir)
        / f"shift_validation_comparison_{source_index}_to_{target_index}_{shift_x:.1f}_{shift_y:.1f}.png"
    )
    plt.close(fig)

    if len(valid_stars) >= 5:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        for i, (x, y, x_shifted, y_shifted) in enumerate(valid_stars[:5]):
            axes[0, i].imshow(simple_scale(source_frame), cmap="viridis", origin="upper")
            axes[0, i].set_xlim(x - box_size * 2, x + box_size * 2)
            axes[0, i].set_ylim(y + box_size * 2, y - box_size * 2)
            axes[0, i].plot(x, y, "g+", markersize=10)
            axes[0, i].add_patch(
                patches.Rectangle(
                    (x - half_box, y - half_box),
                    box_size,
                    box_size,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            axes[0, i].set_title(f"Source Star {i}", fontsize=8)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].imshow(simple_scale(target_frame), cmap="viridis", origin="upper")
            axes[1, i].set_xlim(x_shifted - box_size * 2, x_shifted + box_size * 2)
            axes[1, i].set_ylim(y_shifted + box_size * 2, y_shifted - box_size * 2)
            axes[1, i].plot(x_shifted, y_shifted, "g+", markersize=10)
            axes[1, i].add_patch(
                patches.Rectangle(
                    (x_shifted - half_box, y_shifted - half_box),
                    box_size,
                    box_size,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            axes[1, i].set_title(f"Target Star {i}", fontsize=8)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        plt.tight_layout()
        plt.savefig(
            Path(settings.plotting.output_dir)
            / f"shift_validation_closeups_{source_index}_to_{target_index}_{shift_x:.1f}_{shift_y:.1f}.png"
        )
        plt.close(fig)

    if len(source_fluxes) > 1:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(source_fluxes, target_fluxes, alpha=0.7)
        ax.set_xlabel("Source Flux")
        ax.set_ylabel("Target Flux")
        ax.set_title(
            f"Flux Correlation: {correlation:.3f}, Ratio: {median_ratio:.3f}±{ratio_std:.3f}"
        )

        max_flux = max(np.max(source_fluxes), 1)
        x_line = np.linspace(0, max_flux, 100)
        ax.plot(x_line, median_ratio * x_line, "r--", label=f"Median Ratio: {median_ratio:.3f}")
        ax.legend()

        plt.savefig(
            Path(settings.plotting.output_dir)
            / f"shift_validation_correlation_{source_index}_to_{target_index}_{shift_x:.1f}_{shift_y:.1f}.png"
        )
        plt.close(fig)


def plot_limiting_magnitude(
    filtered_magnitudes: np.ndarray,
    filtered_log_snrs: np.ndarray,
    weights: np.ndarray,
    slope: float,
    intercept: float,
    min_snr: float,
    limiting_mag: float,
    frame_index: int,
) -> None:
    """Render and save a limiting-magnitude estimation diagnostic plot.

    Args:
        filtered_magnitudes (np.ndarray): Star magnitudes used in the fit.
        filtered_log_snrs (np.ndarray): Base-10 log SNR values for the stars.
        weights (np.ndarray): Per-star weights used to size scatter markers.
        slope (float): Slope of the fitted magnitude-vs-log(SNR) trend.
        intercept (float): Intercept of the fitted trend.
        min_snr (float): SNR threshold used to define the limiting magnitude.
        limiting_mag (float): Estimated limiting magnitude.
        frame_index (int): Index of the frame, used in the output filename.

    Returns:
        None
    """
    _fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        filtered_magnitudes,
        filtered_log_snrs,
        c="blue",
        alpha=0.5,
        s=weights * 20 / np.max(weights),
        label="Stars",
    )

    mag_range = np.linspace(min(filtered_magnitudes), max(filtered_magnitudes) + 2, 100)
    fitted_line = slope * mag_range + intercept
    ax.plot(mag_range, fitted_line, "r--", label="Fitted Trend")

    ax.axhline(y=np.log10(min_snr), color="g", linestyle=":", label=f"SNR={min_snr} Threshold")
    ax.axvline(x=limiting_mag, color="k", linestyle="--", label="Limiting Magnitude")

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("log10(SNR)")
    ax.set_title("Limiting Magnitude Estimation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig(Path(settings.plotting.output_dir) / f"frame_{frame_index}_limiting_mag.png")
    plt.close()
