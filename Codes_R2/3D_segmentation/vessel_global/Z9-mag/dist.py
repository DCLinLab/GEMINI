import numpy as np
from sklearn.neighbors import KDTree
import yaml
from pathlib import Path
import nrrd
import pandas as pd
from CrystalTracer3D.io import CrystalReader
from skimage.measure import regionprops


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    in_img = r"Y:\All Data\Lab Members\Jiaxi Lu\02_Processed Data\20250505_ReviewData\3_invivo_NFkB_recording\3_3_SpatialRecording\WholeSlice_Nude32IHC\z9highmag_Nude34_Z9_2025_05_15__02_30_42-Create Image Subset-48.czi"
    reader = CrystalReader(in_img)
    repo = Path(r'D:\Zuohan\Z9-mag')
    res = np.array(reader.resolution)
    crystal, _ = nrrd.read(str(repo / 'crystal.nrrd'))
    vessel, _ = nrrd.read(str(repo / 'vessel.nrrd'))
    coords = np.argwhere(vessel > 0) * res
    tree = KDTree(coords)
    df = pd.read_csv(repo / 'intensity.csv')
    c1, c2 = cfg['crystal']['int_chan']

    ct = [region.centroid * res for region in regionprops(crystal)]
    dists, _ = tree.query(ct)
    df['ratio'] = df[c1] / df[c2]
    df['dist2vessel'] = dists.reshape(-1)
    df.to_csv(repo / 'crystal_info.csv', index=False)