import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib.concurrent import process_map

from pathlib import Path
import nrrd
from skimage.io import imsave

import numpy as np
from skimage.transform import resize

from CrystalTracer3D.io import CrystalReader
from CrystalTracer3D.util import stitch_tiles


def _import_tile(args):
    path, ves_chan = args
    reader = CrystalReader(path)
    return reader.read(reader.find_channel(ves_chan)).max(axis=0, keepdims=True)

def _import_pos(path):
    return CrystalReader(path).coord()

raw_dir = Path(r'Y:\All Data\Lab Members\Jiaxi Lu\01_Raw Data\Tickertape\2025-26\Cancer_GEAR\InVivo\20250324_NudeMice26-33_LPSDoseDependency_Snap_LSM780\Nude32_CD31-IHC\10x_Scan')

def main(path):
    imgs = [i for i in (raw_dir / path.parent.name).glob('*.czi') if 'Shading' not in i.name]
    imgs = sorted(imgs)
    arglist = [(i, 'Ch2-T2') for i in imgs]
    tiles = process_map(_import_tile, arglist)
    pos = process_map(_import_pos, imgs)
    whole = stitch_tiles(tiles, pos)[0]
    imsave(path.parent / 'ves_raw.png', whole)


if __name__ == '__main__':

    indir = Path(r'D:\Zuohan\vessel_batch')
    paths = list(indir.rglob('ves_seg.nrrd'))
    for i in paths:
        main(i)
