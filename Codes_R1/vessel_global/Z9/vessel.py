import yaml
from tqdm.contrib.concurrent import process_map
from skimage.util import img_as_ubyte
from skimage.io import imsave
import nrrd
from skimage.transform import rescale, resize

from CrystalTracer3D.segment import *
from CrystalTracer3D.util import *
from CrystalTracer3D.plot import *


with open('config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def import_tile(path):
    reader = CrystalReader(path)
    return reader.read(reader.find_channel(cfg['vessel']['chan']))

def import_pos(path):
    reader = CrystalReader(path)
    pos = reader.coord()
    return pos


if __name__ == '__main__':
    indir = Path(cfg['paths']['image'])
    outdir = indir.parent
    imgs = sorted(indir.glob('*(*).czi'))

    # load data, we don't need to filter as we first assemble and then segment
    # the tiles are mip but are still 3D
    tiles = process_map(import_tile, imgs)
    pos = process_map(import_pos, imgs)

    # assebmle
    whole = stitch_tiles(tiles, pos)

    # segment vessels
    sf = cfg['vessel']['seg_scale']
    scaled = rescale(whole, (1, sf, sf), anti_aliasing=True)
    seg = segment_vessel(scaled)
    seg = resize(seg, whole.shape)
    header = {
        'type': 'unsigned char',
        'dimension': 3,
        'sizes': list(seg.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(outdir / 'ves_seg.nrrd'), img_as_ubyte(seg), header)
    imsave(outdir / 'ves_seg_mip.png', img_as_ubyte(seg.max(axis=0)))
