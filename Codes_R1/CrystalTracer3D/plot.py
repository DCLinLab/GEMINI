from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from pathlib import Path

from .io import CrystalReader
from skimage.color import label2rgb
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter


def plot_heatmap_mip(data: np.ndarray, cmap='seismic', log=True, colorbar_range=None):
    cmap = cm.get_cmap(cmap).with_extremes(bad='k')

    if data.ndim == 3:
        data = data.max(axis=0)
    mask = data == -1
    data = np.ma.array(data, mask=mask)

    fig, ax = plt.subplots(facecolor='black', figsize=(12, 10))
    plt.style.use('dark_background')
    if log:
        if colorbar_range is not None:
            norm = colors.LogNorm(vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            norm = colors.LogNorm(vmin=max(data.min(), 1e-2), vmax=data.max())
        plt.imshow(data, cmap=cmap, interpolation='nearest', norm=norm)
    else:
        data = data / data.max()
        if colorbar_range is not None:
            norm = colors.Normalize(vmin=colorbar_range[0], vmax=colorbar_range[1])
            plt.imshow(data, cmap=cmap, interpolation='nearest', norm=norm)
        else:
            plt.imshow(data, cmap=cmap, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    cbar = plt.colorbar(aspect=30)
    formatter = LogFormatter(10.0, False, (5, 1))
    cbar.ax.yaxis.set_minor_formatter(formatter)


def plot_label_snapshot(labels):
    proj = np.array(labels).max(axis=0)
    proj = label2rgb(proj)
    plt.imshow(proj)
    plt.axis('off')


def colorize(im, color, clip_percentile=0.1):
    im_scaled = im.astype(np.float32) - np.percentile(im, clip_percentile)
    im_scaled = im_scaled / np.percentile(im_scaled, 100 - clip_percentile)
    im_scaled = np.clip(im_scaled, 0, 1)
    im_scaled = np.atleast_3d(im_scaled)
    color = np.asarray(color).reshape((1, 1, -1))
    return im_scaled * color


def multi_channel_to_rgb(im, palette='tab10', clip_percentile=0.1):
    num_channels = im.shape[0]
    colors = sns.color_palette(palette, num_channels)
    composite = np.zeros((im.shape[1], im.shape[2], 3), dtype=np.float32)
    for i in range(num_channels):
        colored_channel = colorize(im[i, ...], colors[i], clip_percentile)
        composite += colored_channel
    return np.clip(composite, 0, 1)


def plot_folded_profiles(out_dir, analyzer, palette='deep'):
    """
    Plot and save intensity profiles for crystal centers.

    This function generates plots for intensity profiles corresponding to given crystal centers
    and saves each plot as a PNG file in the specified output directory. Each profile is plotted
    with channels in the order specified by `channel_order` and colored using the specified seaborn
    palette.

    Parameters:
        out_dir (Path or str): The directory where the generated plots will be saved.
        profiles (np.ndarray): Intensity profiles with shape (channels, profiles, profile_length).
        channel_order (list or tuple, optional): A list or tuple specifying the order of channels to be plotted.
                                                 Defaults to all channels (i.e. range(profiles.shape[0])).
        palette (str, optional): Name of the seaborn palette to use for the colors. Defaults to 'deep'.

    Returns:
        None

    Notes:
        - The `out_dir` should be a valid directory path. If it does not exist, it will be created.
    """
    profiles = analyzer.aligned_profiles.transpose([1, 0, 2])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If channel_order is not provided, use all channels
    channel_order = list(range(profiles.shape[0]))

    num_channels = len(channel_order)

    # Get a list of colors from the chosen seaborn palette
    colors = sns.color_palette(palette, num_channels)

    ct = profiles.shape[2] // 2
    for i in range(profiles.shape[1]):
        p = profiles[:, i, :]
        p1 = p[:, :ct][:, ::-1]
        p2 = p[:, ct:]
        for idx, ch in enumerate(channel_order):
            plt.plot(p1[ch], color=colors[idx], linestyle='-')
            plt.plot(p2[ch], color=colors[idx], linestyle='--')

        # Create custom legend handles
        solid_line = Line2D([0], [0], color='black', linestyle='-')
        dashed_line = Line2D([0], [0], color='black', linestyle='--')
        # Add legend with custom handles
        plt.legend([solid_line, dashed_line], ['Left', 'Right'])

        plt.title(f'Profile {i}')
        plt.ylabel("Intensity")
        plt.xlabel("Profile Index")

        # Center the x-axis at zero
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.tight_layout()
        plt.savefig(out_dir / f"{i}.png", dpi=300)
        plt.close()


def plot_profiles(out_dir, analyzer, palette='deep'):
    """
    Plot and save intensity profiles for crystal centers.

    This function generates plots for intensity profiles corresponding to given crystal centers
    and saves each plot as a PNG file in the specified output directory. Each profile is plotted
    with channels in the order specified by `channel_order` and colored using the specified seaborn
    palette.

    Parameters:
        out_dir (Path or str): The directory where the generated plots will be saved.
        profiles (np.ndarray): Intensity profiles with shape (channels, profiles, profile_length).
        channel_order (list or tuple, optional): A list or tuple specifying the order of channels to be plotted.
                                                 Defaults to all channels (i.e. range(profiles.shape[0])).
        palette (str, optional): Name of the seaborn palette to use for the colors. Defaults to 'deep'.

    Returns:
        None

    Notes:
        - The `out_dir` should be a valid directory path. If it does not exist, it will be created.
    """
    profiles = analyzer.profiles
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_values = np.arange(analyzer.num_samples) - analyzer.num_samples // 2
    colors = sns.color_palette(palette, profiles.shape[1])
    for i, p in enumerate(profiles):
        for j, ch in enumerate(p):
            plt.plot(x_values * analyzer.unit_dist[i], ch, color=colors[j])
        plt.title(f'Profile {i}')
        plt.ylabel("Intensity")
        plt.xlabel("Distance (um)")

        # Center the x-axis at zero
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.tight_layout()
        plt.savefig(out_dir / f"{i}.png", dpi=300)
        plt.close()


def plot_measurement(out_dir, analyzer, reader: CrystalReader, palette='deep'):
    """
    Draw rotated boxes on 2D image crops, display them using matplotlib, and save the figures.

    For each crop in the input list, this function computes a rotated rectangle (box) that is centered
    on the crop's center. The box is defined by a specified length (long side) and width (short side) and is
    oriented so that its long side aligns with the provided 2D direction vector. Each crop is first converted
    to an 8-bit image, then displayed with
    the rotated box overlayed using matplotlib. The figure for each crop is saved as a PNG file in the given
    output directory.

    Parameters:
        out_dir (str or pathlib.Path): Directory where the output PNG files will be saved.
    Returns:
        None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diam = round(analyzer.max_diam * 1.2)
    ct = diam // 2
    hl = analyzer.num_samples / 2
    hw = analyzer.width / 2

    for i, (center, direction) in enumerate(zip(analyzer.centers, analyzer.directions)):
        crop = [reader.crop(ch, int(center[0]), center[1:], diam, diam) for ch in analyzer.channels]
        crop = np.stack(crop)

        # Half dimensions of the box
        step_w = np.array([-direction[1], direction[0]]) * hw
        step_l = direction * hl

        # Compute the corners of the rotated box (in (row, col) = (y, x) order)
        top_left     = ct - step_l[0] - step_w[0], ct - step_l[1] - step_w[1]
        top_right    = ct - step_l[0] + step_w[0], ct - step_l[1] + step_w[1]
        bottom_right = ct + step_l[0] + step_w[0], ct + step_l[1] + step_w[1]
        bottom_left  = ct + step_l[0] - step_w[0], ct + step_l[1] - step_w[1]
        corners = np.array([top_left, top_right, bottom_right, bottom_left])

        # Convert the crop to an 8-bit image with channels reordered per bgr_index
        crop = multi_channel_to_rgb(crop, palette)

        # Create a matplotlib figure
        fig, ax = plt.subplots()
        ax.imshow(crop)

        # Note: matplotlib expects polygon coordinates as (x, y). Since our corners are (row, col) or (y, x),
        # we swap them.
        polygon_coords = corners[:, ::-1]
        polygon = Polygon(polygon_coords, closed=True, fill=False, edgecolor=np.array([1., 1., 1.]), linewidth=2)
        ax.add_patch(polygon)
        ax.axis('off')

        # Save the figure to a file
        plt.savefig(out_dir / f"crop_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def plot_pred_dist(data, prediction='prediction', target='target'):
    # Initialize the FacetGrid object
    g = sns.FacetGrid(data, row=target, hue=target, aspect=10, height=0.5, palette='coolwarm')

    # Map the kdeplots to the FacetGrid
    g.map(sns.kdeplot, prediction,
          bw_adjust=0.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, prediction, clip_on=False, color='w', lw=2, bw_adjust=0.5)
    # Add scatter plots beneath each density plot
    g.map(sns.scatterplot, prediction, color='k', s=10, alpha=0.5)
    # Add a horizontal line at the base of each density plot
    g.refline(y=0, linewidth=2, linestyle='-', clip_on=False)

    # Define a function to label each plot
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight='bold', color=color,
                ha='left', va='center', transform=ax.transAxes)

    g.map(label, prediction)

    # Adjust the subplots to overlap
    g.figure.subplots_adjust(hspace=0)

    # Remove axes details that don't play well with overlap
    g.set_titles('')
    g.set(yticks=[], ylabel='')
    g.despine(bottom=True, left=True)