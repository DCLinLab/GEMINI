import yaml
import nrrd
from tqdm.contrib.concurrent import process_map

from CrystalTracer3D.io import *
from CrystalTracer3D.util import *
from CrystalTracer3D.plot import *


with open('config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def take(args):
    lut, path = args
    lut = [0] + list(lut)       # label starts from 1
    indices = np.load(path / 'crystal.npy')
    tile = np.take(lut, indices)
    return tile


if __name__ == '__main__':
    rawdir = Path(cfg['paths']['image'])
    resdir = Path(cfg['paths']['crystal'])
    c1, c2 = cfg['crystal']['int_chan']

    all_pos = [CrystalReader(i).coord() for i in rawdir.glob('*.czi')]
    cr = get_canvas_range(all_pos, 1024, 1024)
    paths = list(resdir.iterdir())

    lut, pos = [], []
    # load the ratio values and set below threshold size ones as 0
    for p in tqdm(paths):
        reader = CrystalReader(rawdir / p.with_suffix('.czi').name)
        pos.append(reader.coord())
        table = pd.read_csv(p / 'intensity.csv')
        table['ratio'] = table[c1] / table[c2]
        table.loc[table['volume'] < cfg["assemble"]["vol_thr"], 'ratio'] = 0
        lut.append(table['ratio'])

    # normalization
    all_vals = np.hstack(lut)
    m1 = np.percentile(all_vals, cfg['assemble']['low_pct'])
    m2 = np.percentile(all_vals, cfg['assemble']['high_pct'])
    lut = [((v.clip(m1, m2) - m1) / (m2 - m1) * 255).round() for v in lut]

    # assign the values onto each segmented tile
    tiles = process_map(take, list(zip(lut, paths)))

    # output
    final = stitch_tiles(tiles, pos, cr)
    header = {
        'type': 'unsigned char',
        'dimension': 3,
        'sizes': list(final.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(resdir.parent / 'ratiometrics.nrrd'), final, header)
    plot_heatmap_mip(final)
    plt.savefig(resdir.parent / 'example.png', dpi=300)
