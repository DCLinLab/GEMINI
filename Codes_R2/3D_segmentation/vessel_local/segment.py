from CrystalTracer3D.segment import *
from CrystalTracer3D.io import *
from CrystalTracer3D.band import *
import yaml
from sklearn.neighbors import KDTree
from tqdm.contrib.concurrent import process_map
from skimage.measure import regionprops
import pandas as pd


with open('../vessel_density_local/config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def main(in_img):
    outdir = Path(cfg['paths']['segmentation']) / in_img.stem
    outdir.mkdir(parents=True, exist_ok=True)
    reader = CrystalReader(in_img)
    vessel_chan = reader.find_channel(cfg['segmentation']['vessel_ch'])
    seg_chan = reader.find_channel(cfg['segmentation']['seg_chan'])
    sf = cfg['segmentation']['xy_scale']

    # segment vessels (whole image, downscaled)
    res = reader.resolution / np.array((1, sf, sf))
    ves = reader.assemble_tiles(vessel_chan, sf)
    seg = segment_vessel(ves)
    coords = np.argwhere(seg > 0) * res      # shape: (N, 3)
    np.save(outdir / 'vessel', seg)
    # building a kdtree to get the nearest voxel
    tree = KDTree(coords)

    # segment crystals (whole crop)
    gfp = reader.assemble_tiles(seg_chan, sf)
    labels = segment_crystal_whole(gfp, res, min_volume=cfg['segmentation']['min_vol'])
    np.save(outdir/ f'crystal', labels)

    # measure the shortest distances to vessels for each crystal
    ct = [region.centroid * res for region in regionprops(labels)]
    dists, _ = tree.query(ct)
    df = pd.DataFrame({'label': np.arange(len(dists)) + 1, 'dist': dists.reshape(-1)})
    df.to_csv(outdir / f'dist2vessel.csv', index=False)


if __name__ == '__main__':
    indir = Path(cfg['paths']['image'])
    imgs = sorted(indir.glob('*.czi'))
    process_map(main, imgs)
