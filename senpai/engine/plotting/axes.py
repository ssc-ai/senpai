"""Matplotlib figure/axes factory helpers sized to match image dimensions."""

import matplotlib

matplotlib.use("Svg")
import matplotlib.pyplot as plt


def prep_axes_plot(width: int, height: int, dpi: int = 75) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure and axes sized to an image in inches at the given DPI.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        dpi: Dots-per-inch used to convert pixel dimensions to inches.

    Returns:
        The created ``(figure, axes)`` pair.
    """
    fig, ax = plt.subplots(1, figsize=(1 * width / dpi, 1 * height / dpi), dpi=dpi)
    return fig, ax


def prep_axes(height: int, width: int, dpi: int = 150) -> tuple[plt.Figure, plt.Axes]:
    """Create a borderless, full-bleed figure and axes for rendering an image.

    The axes fill the entire figure with no frame, ticks, or margins so an
    image drawn into them maps one-to-one onto the output canvas.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        dpi: Dots-per-inch used to convert pixel dimensions to inches.

    Returns:
        The created ``(figure, axes)`` pair.
    """
    fig = plt.figure(frameon=False, figsize=(1 * width / dpi, 1 * height / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    return fig, ax
