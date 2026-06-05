import matplotlib

matplotlib.use("Svg")
import matplotlib.pyplot as plt


def prep_axes_plot(width: int, height: int, dpi: int = 75) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, figsize=(1 * width / dpi, 1 * height / dpi), dpi=dpi)
    return fig, ax


def prep_axes(height: int, width: int, dpi: int = 150) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(frameon=False, figsize=(1 * width / dpi, 1 * height / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    return fig, ax
