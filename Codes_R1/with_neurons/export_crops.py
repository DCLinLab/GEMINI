import nrrd
from pathlib import Path
import yaml
from CrystalTracer3D.io import CrystalReader
import pandas as pd
import numpy as np
from skimage.segmentation import expand_labels
from skimage.measure import regionprops
import napari
from tqdm import tqdm
from scipy.stats import rankdata


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        repo = Path(cfg['data']['segmentation'])
        slice_range = cfg['data']['range']

    lab_n, _ = nrrd.read(str(repo / "neuron.nrrd"))
    lab_c, _ = nrrd.read(str(repo / "crystal.nrrd"))

    reader = CrystalReader(cfg['data']['path'])
    ch = reader.find_channel(cfg['data']['neuron'])
    img_n = reader.read(ch)[slice_range[0]:slice_range[1]]
    # ch = reader.find_channel(cfg['data']['crystal'])
    # img_c = reader.read(ch)[slice_range[0]:slice_range[1]]
    lab_n = expand_labels(lab_n, 1, spacing=reader.resolution)

    neurons = regionprops(lab_n)
    crystals = regionprops(lab_c)
    lab_n_map = {j.label: i for i, j in enumerate(neurons)}
    lab_c_map = {j.label: i for i, j in enumerate(crystals)}

    neurons_df = pd.read_csv(repo / "neuron_prop.csv", index_col='label')
    neurons_df['rank'] = rankdata(neurons_df.index, method='min')
    pairs = pd.read_csv(repo / "pair.csv")
    out_dir = repo / 'crops'
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in tqdm(np.unique(pairs['neuron'])):
        no = neurons_df.at[n, 'rank']
        op = out_dir / f'{no}.png'
        # if op.exists():
        #     continue
        q = pairs.query(f'neuron == {n}')
        sli = neurons[lab_n_map[n]].slice
        c = q['crystal'].to_numpy()
        for i in c:
            temp = crystals[lab_c_map[i]].slice
            sli = tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(sli, temp))
        a = lab_n[sli] == n
        b = np.isin(lab_c[sli], c)
        c = img_n[sli]
        # header = {
        #     'dimension': 3,
        #     'sizes': list(a.shape),
        #     'encoding': 'gzip'
        # }
        # nrrd.write(str(out_dir / f'{n}_neuron.nrrd'), img_as_ubyte(a), header)
        # nrrd.write(str(out_dir / f'{n}_crystal.nrrd'), img_as_ubyte(b), header)
        # nrrd.write(str(out_dir / f'{n}_image.nrrd'), c, header)
        viewer = napari.Viewer()
        viewer.add_labels(b, name='crystal', opacity=.7, colormap={1: 'red', None: [0, 0, 0, 0]}, blending='additive')
        viewer.add_labels(a, name='neuron', opacity=.7, colormap={1: 'green', None: [0, 0, 0, 0]}, blending='additive')
        # layer = viewer.add_image(c, opacity=.7, blending='additive')
        # layer.contrast_limits = [3587.828571428571, 14532.678571428572]
        viewer.add_points(
            [[a.shape[0] / 2, 0, a.shape[2] / 2]],
            features={'no': [no]},
            text={
                'string': '{no}',
                'size': 40,
                'scaling': True,
                'color': 'white',
                'translation': np.array([0, -20, 0]),
            },
            size=0,
            face_color='red',
        )
        viewer.layers[0].iso_gradient_mode = 'smooth'
        viewer.layers[1].iso_gradient_mode = 'smooth'
        viewer.dims.ndisplay = 3
        # viewer.camera.perspective = 45
        viewer.camera.zoom = 2
        viewer.camera.angles = (-1.5, 20, 85)
        viewer.screenshot(path=str(op), scale=4, canvas_only=True)
        viewer.close()

