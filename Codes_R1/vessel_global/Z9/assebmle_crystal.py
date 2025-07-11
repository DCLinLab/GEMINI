import numpy as np
import yaml
import nrrd
from tqdm.contrib.concurrent import process_map

from CrystalTracer3D.io import *
from CrystalTracer3D.util import *
from CrystalTracer3D.plot import *


with open('../config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def take(args):
    lut, path = args
    indices = np.load(path / 'crystal.npy')
    tile = np.take(lut, indices)
    return tile


if __name__ == '__main__':
    rawdir = Path(cfg['paths']['image'])
    resdir = Path(cfg['paths']['crystal'])

    all_pos = [CrystalReader(i).coord() for i in rawdir.glob('*.czi')]
    cr = get_canvas_range(all_pos, 1024, 1024)

    pos = []
    lut = []
    tot = 0
    paths = list(resdir.iterdir())
    # create lookup tables and filter below threshold size ones
    # remap the labels of the whole image
    # 0 will be background, the rest labels will be ordered from the first tile, starting from 1
    # each lookup table must contain a preceding 0 for each tile as background
    for p in tqdm(paths):
        reader = CrystalReader(rawdir / p.with_suffix('.czi').name)
        pos.append(reader.coord())
        table = pd.read_csv(p / 'intensity.csv')
        lookup = np.arange(len(table)) + 1
        lookup[table['volume'] < cfg["assemble"]["vol_thr"]] = 0
        lookup = [0] + list(lookup)
        unq = list(np.unique(lookup))
        lookup = np.array([unq.index(i) for i in lookup])
        lookup[lookup > 0] += tot
        tot = lookup.max()
        lut.append(lookup)

    tiles = process_map(take, list(zip(lut, paths)))

    # output
    final = stitch_tiles(tiles, pos, cr)
    header = {
        'dimension': 3,
        'sizes': list(final.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(resdir.parent / 'crystal_whole.nrrd'), final, header)
    plot_label_snapshot(final)
    plt.savefig(resdir.parent / f'crystal_whole.png', dpi=300, bbox_inches='tight', pad_inches=0)
