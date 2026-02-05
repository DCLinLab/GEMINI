import yaml
from pathlib import Path
from skimage.io import imread, imsave
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import rpack


def crop(img):
    mask = img[..., :3].max(axis=-1)
    nz = np.nonzero(mask > 0)
    y_min, y_max = np.min(nz[0]), np.max(nz[0])
    x_min, x_max = np.min(nz[1]), np.max(nz[1])
    # 注意索引上限要 +1
    return img[y_min:y_max + 1, x_min:x_max + 1]


def set_transparent(img):
    img[..., -1] = (img[..., :-1].max(axis=-1) > 0) * 255


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        repo = Path(cfg['data']['segmentation'])
    indir = repo / 'crops'
    paths = natsorted(indir.glob('*.png'))
    imgs = []
    odir = repo / 'tranparent_crops'
    odir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(paths):
        n = int(i.stem)
        if n in cfg['data']['drop_neuron']:
            continue
        op = odir / i.name
        i = imread(i)
        i = crop(i)
        set_transparent(i)
        imsave(op, i)
        imgs.append(i)

    # imgs = np.array(imgs)
    # canvas = montage(imgs, channel_axis=-1)

    sizes = [(i.shape[1], i.shape[0]) for i in imgs]
    r1 = round(len(imgs) ** .5 * max([i.shape[1] for i in imgs]))
    r2 = round(len(imgs) ** .5 * max([i.shape[0] for i in imgs]))
    positions = rpack.pack(sizes, max_width=r1, max_height=r2)
    canvas_w, canvas_h = rpack.bbox_size(sizes, positions)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=imgs[0].dtype)

    def paste(img, x, y):
        h, w = img.shape[:2]
        # 将左下角 (x, y) 转成 NumPy 数组的索引：
        canvas[canvas_h - y - h: canvas_h - y, x: x + w, :] = img

    for img, (w, h), (x, y) in zip(imgs, sizes, positions):
        paste(img, x, y)

    imsave(repo / 'tiled_crops.png', canvas)