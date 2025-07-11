from pathlib import Path
import nrrd
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map


def main(i):
    img, _ = nrrd.read(str(i))
    img = img.max(axis=0)
    height, width = img.shape[:2]
    dpi = 300
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(facecolor='black', figsize=figsize, dpi=dpi)
    plt.style.use('dark_background')
    plt.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    plt.savefig(i.with_name('ves_mip2.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    indir = Path(r'D:\Zuohan\vessel_batch')
    args = list(indir.rglob('ves_seg.nrrd'))
    process_map(main, args)
