import matplotlib.pyplot as plt
import seaborn as sns
from skimage.util import img_as_ubyte
from tqdm.contrib.concurrent import process_map

from pathlib import Path
import nrrd
import pandas as pd

cols = 100


def split_tiles_cf(img):
    C, H, W = img.shape
    rows = round(H/W*cols)
    th, tw = H // rows, W // cols
    img = img[:, :th*rows, :tw*cols]  # 裁切到可整除范围
    tiles = img.reshape(C,
                        rows, th,
                        cols, tw)  # shape -> (C, rows, th, cols, tw)
    tiles = tiles.transpose(1, 3, 0, 2, 4)
    # final shape: (rows, cols, C, th, tw)
    return tiles

import numpy as np
from skimage.transform import resize, rescale
from skimage.io import imread
import matplotlib.colors as colors
from matplotlib import cm

sf = 0.2


def vessel(path):
    img = imread(path.parent / 'ves_raw.png')
    img = rescale(img, sf)
    img = ((img - img.mean()) / img.std()).clip(-1, 1)
    img = (img - img.min()) / (img.max() - img.min())
    seg, _ = nrrd.read(str(path))
    tiles = split_tiles_cf(seg > 0)
    tiles = tiles.mean(axis=(2, 3, 4))
    a = resize(tiles, img.shape, order=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    norm = colors.LogNorm(vmin=1e-2, vmax=0.1)
    cmap = cm.get_cmap('seismic').with_extremes(bad='k')
    plt.imshow(a, cmap=cmap, interpolation='nearest', norm=norm)
    cbar = plt.colorbar(aspect=30)
    plt.imshow(img, cmap='gray', alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path.parent / 'ves_density.png', dpi=300)
    # plt.show()
    plt.close()
    return tiles


def ratio(path):
    img = imread(path.parent / 'ves_raw.png')
    img = rescale(img, sf)
    img = ((img - img.mean()) / img.std()).clip(-1, 1)
    img = (img - img.min()) / (img.max() - img.min())
    ratio, _ = nrrd.read(str(path.parent / 'ratiometrics.nrrd'))
    tiles = split_tiles_cf(ratio)
    tiles = tiles.mean(axis=(2, 3, 4), where=(tiles > 0))
    a = resize(tiles, img.shape, order=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    norm = colors.LogNorm(vmin=1e-2, vmax=10)
    cmap = cm.get_cmap('seismic').with_extremes(bad='k')
    plt.imshow(a, cmap=cmap, interpolation='nearest', norm=norm)
    cbar = plt.colorbar(aspect=30)
    plt.imshow(img, cmap='gray', alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path.parent / 'ratiometrics2.png', dpi=300)
    # plt.show()
    plt.close()
    return tiles



def main(path):
    t1 = vessel(path).reshape(-1)
    t2 = ratio(path).reshape(-1)
    H, W = t1.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    data = {
        'x': xs.reshape(-1),
        'y': ys.reshape(-1),
        'vessel': t1.reshape(-1),
        'ratio': t2.reshape(-1)
    }
    df = pd.DataFrame(data)
    df = df[~df.isna().any(axis=1)]
    df.to_csv(path.parent / 'ves_ratio.csv', index=False)

    max_idx = np.unravel_index(np.nanargmax(ratio), ratio.shape)
    max_y, max_x = max_idx



if __name__ == '__main__':

    indir = Path(r'D:\Zuohan\vessel_batch')
    paths = list(indir.rglob('ves_seg.nrrd'))


    process_map(main, paths, max_workers=1)
