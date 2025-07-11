import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.util import img_as_ubyte
import nrrd
from skimage.measure import regionprops

from CrystalTracer3D.io import CrystalReader
from CrystalTracer3D.segment import segment_filled_crystal
from CrystalTracer3D.plot import plot_label_snapshot


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    in_img = Path(cfg['data']['path'])
    out_dir = Path(cfg['data']['segmentation'])
    seg_chan = cfg['data']['crystal']
    slice_range = cfg['data']['range']

    reader = CrystalReader(in_img)
    ch = reader.find_channel(seg_chan)
    img = reader.read(ch)[slice_range[0]:slice_range[1]]

    # segment crystals
    res = reader.resolution
    seg = segment_filled_crystal(img, **cfg['crystal'], zspacing=res[0] / res[1])
    crystals = regionprops(seg, spacing=res)
    labels = np.array([i.label for i in crystals])
    vol = np.array([i.area for i in crystals])
    keep = labels[vol > cfg['filter']['crystal_volume']]

    new = np.zeros_like(seg)
    for lab in keep:
        new[seg == lab] = lab

    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        'dimension': 3,
        'sizes': list(new.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(out_dir / 'crystal.nrrd'), new, header)
    imsave(out_dir / 'crystal_raw.png', img_as_ubyte(img).max(axis=0))
    plot_label_snapshot(new)
    plt.savefig(out_dir / 'crystal.png', dpi=300, bbox_inches='tight', pad_inches=0)