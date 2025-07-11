import numpy as np
import yaml
from pathlib import Path
import nrrd
from skimage.io import imsave
from skimage.util import img_as_ubyte
from CrystalTracer3D.plot import plot_label_snapshot
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
import pandas as pd
from CrystalTracer3D.io import CrystalReader
from CrystalTracer3D.segment import segment_soma
from tqdm import tqdm


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    in_img = Path(cfg['data']['path'])
    out_dir = Path(cfg['data']['segmentation'])
    slice_range = cfg['data']['range']
    out_dir.mkdir(parents=True, exist_ok=True)

    # segment crystals
    reader = CrystalReader(in_img)
    ch = reader.find_channel(cfg['data']['neuron'])
    img = reader.read(ch)[slice_range[0]:slice_range[1]]
    # ch = reader.find_channel(cfg['data']['crystal'])
    # img += reader.read(ch)[slice_range[0]:slice_range[1]]

    res = reader.resolution
    seg = segment_soma(img, **cfg['neuron'], zspacing=res[0] / res[1])
    neurons = regionprops(seg, spacing=res)
    labels = np.array([i.label for i in neurons])
    # conv = process_map(convex_hull_image, [i.image for i in with_neurons])
    true_ct = []
    radius = []
    for i in tqdm(neurons, 'Postprocessing', total=len(neurons)):
        dt = distance_transform_edt(i.image_filled.max(axis=0), res[1:])
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        r = dt[y, x]

        dt = distance_transform_edt(i.image_filled.max(axis=2), res[:2])
        z, _ = np.unravel_index(np.argmax(dt), dt.shape)

        z += i.slice[0].start
        y += i.slice[1].start
        x += i.slice[2].start
        radius.append(r)
        true_ct.append([z, y, x])
    radius = np.array(radius)
    true_ct = np.array(true_ct)
    t = radius >= cfg['filter']['neuron_radius']
    pd.DataFrame({'radius': radius[t], 'z': true_ct[t, 0], 'y': true_ct[t, 1],
                  'x': true_ct[t, 2], 'label': labels[t]}).to_csv(out_dir / 'neuron_prop.csv', index=False)
    new = np.zeros_like(seg)
    for lab in labels[t]:
        new[seg == lab] = lab

    header = {
        'dimension': 3,
        'sizes': list(new.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(out_dir / 'neuron.nrrd'), new, header)
    imsave(out_dir / 'neuron_raw.png', img_as_ubyte(img).max(axis=0))
    plot_label_snapshot(new)
    plt.savefig(out_dir / 'neuron.png', dpi=300, bbox_inches='tight', pad_inches=0)
