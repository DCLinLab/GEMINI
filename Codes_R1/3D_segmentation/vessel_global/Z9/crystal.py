import yaml
from skimage.transform import rescale
from tqdm.contrib.concurrent import process_map
from skimage.io import imsave
from skimage.util import img_as_ubyte

from CrystalTracer3D.intensity import *
from CrystalTracer3D.io import *
from CrystalTracer3D.segment import *
from CrystalTracer3D.util import *
from CrystalTracer3D.plot import *


with open('config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def main(in_img):
    outdir = Path(cfg['paths']['crystal']) / in_img.stem
    reader = CrystalReader(in_img)
    seg_chan = reader.find_channel(cfg['crystal']['seg_chan'])

    # segment crystals
    img = reader.read(seg_chan)
    if snr(img) < cfg['crystal']['snr_thresh']:
        print(f'Skipping {in_img} for low SNR')
        return
    labels = segment_filled_crystal(img, gamma=.7)

    # save files
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / 'crystal', labels)     # the label file
    imsave(outdir / 'raw_mip.png', img_as_ubyte(img).max(axis=0))
    plot_label_snapshot(labels)
    plt.savefig(outdir / f'crystal.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # compute the grayscale of different channels and export as tables
    a = SegmentationAnalyzer(labels, reader.resolution)
    table = {}
    for i in cfg['crystal']['int_chan']:
        chan = reader.find_channel(i)
        img = reader.read(chan)
        col = a.extract_avg_intensity(img)
        table[i] = col
    assert 'volume' not in table
    table['volume'] = a.volume
    table = pd.DataFrame(table)
    table.to_csv(outdir / f'intensity.csv', index=False)


if __name__ == '__main__':
    indir = Path(cfg['paths']['image'])
    imgs = sorted(indir.glob('*.czi'))      # tile by tile storage, each is 1024x1024 & containing many crystalsi
    process_map(main, imgs)
