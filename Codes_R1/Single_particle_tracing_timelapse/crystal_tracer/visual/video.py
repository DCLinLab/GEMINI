from crystal_tracer.visual.draw import draw_contour
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from crystal_tracer.utils import load_czi_slice, get_czi_shape
import platform
import matplotlib.animation as animation
import matplotlib.pyplot as plt


font = cv2.FONT_HERSHEY_PLAIN
color = (255, 255, 255)  # white color
font_scale = 1
position = (2, 2 + int(font_scale * 10))  # position of text, adjust as needed
thickness = 1


def make_video(track: list[tuple[int, int]], save_path: Path | None, czi_path: Path, table_paths: list[Path],
               mask_paths: list[Path], win_rad=30, frame_rate=25., max_time=0., channel=0):
    """
    generate a video for one traced track.

    :param track: a list of tuples containing (frame_index, crystal_id)
    :param save_path: the save path of the AVI video
    :param czi_path: the CZI image path
    :param table_paths: the paths to the crystal statistics in each time frame
    :param mask_paths: the paths to the segmentation of crystals
    :param win_rad: window radius of the video
    :param frame_rate: frame rate of the video
    :param max_time: the maximum time of the video, if 0, no limit. in minute
    """
    assert len(table_paths) == len(mask_paths)
    tag = 'DIVX' if platform.system() == 'Windows' else 'XVID'
    out_size = (win_rad * 2, win_rad * 2)
    writer = None if save_path is None else cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*tag), frame_rate, out_size)
    last = None
    tot, c, height, width, interval = get_czi_shape(czi_path)
    out = []
    elapse = 0.
    for i, j in track:
        ys_old, xs_old, intensity = pd.read_csv(table_paths[i]).loc[j, ['y_start', 'x_start', 'intensity']].values.astype(int).ravel()
        mask = np.load(mask_paths[i])[f'arr_{j}']
        size_y, size_x = mask.shape
        cty, ctx = ys_old + size_y // 2, xs_old + size_x // 2
        ys, ye = max(cty - win_rad, 0), min(height, cty + win_rad)
        xs, xe = max(ctx - win_rad, 0), min(width, ctx + win_rad)
        new_mask = np.zeros([ye - ys, xe - xs], dtype=np.uint8)
        y_, x_ = np.nonzero(mask)
        y_ += ys_old - ys
        x_ += xs_old - xs
        y_ = np.clip(y_, 0, height - 1)
        x_ = np.clip(x_, 0, width - 1)
        new_mask[(y_, x_)] = 1
        img = load_czi_slice(czi_path, 1, i)[ys: ye, xs: xe]
        img = img.clip(None, intensity * 2) / (intensity * 2)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = draw_contour(img, new_mask)
        pad_width = (*out_size, 3) - np.array(img.shape)
        pad_width = np.stack((pad_width // 2, pad_width - pad_width // 2), axis=1)
        img = np.pad(img, pad_width)

        hours, minutes = divmod(elapse, 60)
        minutes, seconds = divmod(minutes, 1)
        text = "{}:{:02d}".format(int(hours), int(minutes))
        # text = f'{elapse / 60:.1f}hr'

        img = cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        for k in range(1 if last is None else i - last):
            out.append(img)
            if writer is not None:
                writer.write(img)
            elapse += interval
            if elapse > max_time > 0:
                break
        if elapse > max_time > 0:
            break
        last = i
    return out


def normalized_volume_growth(table: pd.DataFrame, outpath, fps=30, resample=True, unit='min', normalize=True):
    time = table['time'].to_numpy()
    size = table['area'].to_numpy()
    # turn area to normalized volume
    if normalize:
        size = size ** (3/2)
        size -= size[0]
        size /= size[-1]

    # interpolation to smooth the animation
    if resample:
        stamps = table['timestamp'].to_numpy()
        time = np.linspace(time[0], time[-1], stamps[-1] - stamps[0] + 1)
        size = np.interp(time, table['time'], size)

    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel(f'Time Elapse ({unit})')
    ax.set_ylabel('Volume Index (normalized)')
    ax.set_xlim(0, max(time))
    if normalize:
        ax.set_ylim(0, max(size) * 1.25)
    else:
        ax.set_ylim(0, 1)
    plt.yticks(np.linspace(0, 1., 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect(1 / ax.get_data_ratio(), adjustable='box')

    def update(i):
        line.set_data(time[:i], size[:i])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(time), interval=1000 / fps, blit=True, repeat=False)
    ani.save(Path(outpath).with_suffix('.mp4'), writer='ffmpeg', fps=fps)
