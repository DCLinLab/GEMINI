import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import nrrd
from skimage.measure import regionprops
from sklearn.neighbors import KDTree
from CrystalTracer3D.io import CrystalReader
from tqdm import tqdm


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        repo = Path(cfg['data']['segmentation'])

    reader = CrystalReader(cfg['data']['path'])
    res = np.array(reader.resolution)
    crystal, _ = nrrd.read(str(repo / 'crystal.nrrd'))
    neuron, _ = nrrd.read(str(repo / 'neuron.nrrd'))

    crystals = regionprops(crystal, spacing=res)
    neurons = regionprops(neuron, spacing=res)

    crystal_centers = [i.centroid for i in crystals]
    labels = [i.label for i in neurons]

    df = pd.read_csv(repo / 'neuron_prop.csv', index_col='label')
    neuron_centers = df.loc[labels, ['z', 'y', 'x']].to_numpy() * res
    radius = df.loc[labels, ['radius']].to_numpy()

    tree = KDTree(neuron_centers)
    dists, inds = tree.query(crystal_centers, k=1, return_distance=True)

    pair = []
    far = 0
    for j, (d, i) in tqdm(enumerate(zip(dists, inds)), total=len(dists)):
        d, i = d[0], i[0]
        r1 = radius[i]
        r2 = crystals[j].equivalent_diameter_area / 2
        if d > min(cfg['match']['dist_thr'], r1 + r2):
            far += 1
            continue
        pair.append([crystals[j].label, neurons[i].label, d])
    df = pd.DataFrame(pair, columns=['crystal', 'neuron', 'dist'])
    df = df.loc[df.groupby('crystal')['dist'].idxmin()]
    df.to_csv(repo / 'pair.csv', index=False)
    print(f'too far: {far}')