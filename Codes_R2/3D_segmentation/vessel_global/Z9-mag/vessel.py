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


if __name__ == '__main__':
    in_img = r"Y:\All Data\Lab Members\Jiaxi Lu\02_Processed Data\20250505_ReviewData\3_invivo_NFkB_recording\3_3_SpatialRecording\WholeSlice_Nude32IHC\z9highmag_Nude34_Z9_2025_05_15__02_30_42-Create Image Subset-48.czi"
    outdir = Path(r'D:\Zuohan\Z9-mag')
    reader = CrystalReader(in_img)
    ves_chan = reader.find_channel(cfg['vessel']['chan'])
    img = reader.assemble_tiles(ves_chan)
    seg = segment_vessel(img, thr=(2, 5), sigma=3)
    outdir.mkdir(parents=True, exist_ok=True)
    seg = img_as_ubyte(seg)
    header = {
        'type': 'unsigned char',
        'dimension': 3,
        'sizes': list(seg.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(outdir / 'vessel.nrrd'), seg, header)
    imsave(outdir / 'vessel_raw.png', img_as_ubyte(img).max(axis=0))
    imsave(outdir / 'vessel.png', seg.max(axis=0))

