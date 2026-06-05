"""Diagnostic plots for variable-kernel WCS refinement."""

import logging

import numpy as np

from senpai.core.config import get_config
from senpai.engine.detection.jacobian import get_local_streak_kernel
from senpai.engine.models.senpai import RateTrackFrame

logger = logging.getLogger(__name__)


def plot_variable_kernel_grid(
    frame: RateTrackFrame,
    wcs,
    nx: int = 4,
    ny: int = 4,
) -> None:
    """Diagnostic: plot a grid of local streak kernels across the field of view."""
    from matplotlib import pyplot as plt

    config = get_config()
    if not config.plotting.debug or frame.streak is None:
        return

    height, width = frame.frame.data.shape
    xs = np.linspace(0, width - 1, max(1, nx))
    ys = np.linspace(0, height - 1, max(1, ny))

    fig, axes = plt.subplots(len(ys), len(xs), figsize=(3 * len(xs), 3 * len(ys)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            ax = axes[iy, ix]
            try:
                kernel = get_local_streak_kernel(
                    wcs,
                    frame.streak,
                    x=float(x),
                    y=float(y),
                    scale_width=True,
                    upsample=100,
                    halo_fwhm=None,
                    halo_level=1e-3,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(
                    "Failed to build local streak kernel for grid point (%.1f, %.1f) in frame %d: %s",
                    x,
                    y,
                    frame.index,
                    e,
                )
                ax.set_axis_off()
                continue

            ax.imshow(kernel, origin="lower", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({int(x)}, {int(y)})", fontsize=8)

    fig.suptitle(f"Variable streak kernels - frame {frame.index}", fontsize=10)
    fig.tight_layout()

    output_path = config.runtime.output_dir / f"{frame.index}_variable_kernel_grid.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved variable-kernel grid diagnostic to %s", output_path)


def plot_variable_kernel_star_diagnostic(
    frame: RateTrackFrame,
    image_cutout: np.ndarray,
    kernel: np.ndarray,
    correlation: np.ndarray,
    x_min: int,
    y_min: int,
    measured_x: float,
    measured_y: float,
    star_index: int,
) -> None:
    """Diagnostic: plot image cutout, local kernel, and correlation with peak marked."""
    from matplotlib import pyplot as plt

    config = get_config()
    if not config.plotting.debug:
        return

    ih, iw = image_cutout.shape
    kh, kw = kernel.shape

    # Pad or crop kernel to match cutout size for visualization
    kernel_vis = np.zeros_like(image_cutout)
    if kh <= ih and kw <= iw:
        y0 = (ih - kh) // 2
        x0 = (iw - kw) // 2
        kernel_vis[y0 : y0 + kh, x0 : x0 + kw] = kernel
    else:
        ky0 = max(0, (kh - ih) // 2)
        kx0 = max(0, (kw - iw) // 2)
        kernel_vis[:, :] = kernel[ky0 : ky0 + ih, kx0 : kx0 + iw]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(image_cutout, origin="lower", cmap="gray")
    ax0.set_title("Image cutout")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(kernel_vis, origin="lower", cmap="viridis")
    ax1.set_title("Local kernel (scaled)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(correlation, origin="lower", cmap="magma")
    ax2.set_title("Correlation")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Mark the peak position in correlation coordinates
    peak_x = measured_x - x_min
    peak_y = measured_y - y_min
    ax2.plot(peak_x, peak_y, "rx", markersize=8, mew=1.5)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Variable-kernel diagnostic - frame {frame.index}, star {star_index}",
        fontsize=10,
    )
    fig.tight_layout()

    output_path = (
        config.runtime.output_dir
        / f"{frame.index}_variable_kernel_star_{star_index}.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved variable-kernel star diagnostic to %s", output_path)
