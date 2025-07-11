import yaml
from skimage.io import imsave
from skimage.util import img_as_ubyte
import nrrd

from CrystalTracer3D.intensity import *
from CrystalTracer3D.io import *
from CrystalTracer3D.segment import *
from CrystalTracer3D.plot import *


with open('../config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


if __name__ == '__main__':
    in_img = r"Y:\All Data\Lab Members\Jiaxi Lu\02_Processed Data\20250505_ReviewData\3_invivo_NFkB_recording\3_3_SpatialRecording\WholeSlice_Nude32IHC\z9highmag_Nude34_Z9_2025_05_15__02_30_42-Create Image Subset-48.czi"
    outdir = Path(r'D:\Zuohan\Z9-mag')
    reader = CrystalReader(in_img)
    seg_chan = reader.find_channel(cfg['crystal']['seg_chan'])

    # segment crystals
    img = reader.assemble_tiles(seg_chan)
    img2 = stack_hist_matching(img)
    labels = segment_filled_crystal_by_plane(img2)

    # save files
    outdir.mkdir(parents=True, exist_ok=True)
    header = {
        'dimension': 3,
        'sizes': list(labels.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(outdir / 'crystal.nrrd'), labels, header)
    imsave(outdir / 'crystal_raw.png', img_as_ubyte(img).max(axis=0))
    plot_label_snapshot(labels)
    plt.savefig(outdir / f'crystal.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # compute the grayscale of different channels and export as tables
    a = SegmentationAnalyzer(labels, reader.resolution)
    table = {}
    for i in cfg['crystal']['int_chan']:
        chan = reader.find_channel(i)
        img = reader.assemble_tiles(chan)
        img = stack_hist_matching(img)
        col = a.extract_avg_intensity(img)
        table[i] = col
    assert 'volume' not in table
    table['volume'] = a.volume
    table = pd.DataFrame(table)
    table.to_csv(outdir / f'intensity.csv', index=False)

    # # filter small ones
    # lookup = np.arange(len(table)) + 1
    # lookup[table['volume'] < cfg["assemble"]["vol_thr"]] = 0
    # lookup = [0] + list(lookup)
    # unq = list(np.unique(lookup))
    # lookup = np.array([unq.index(i) for i in lookup])
    # labels = np.take(lookup, labels)
    #
    # header = {
    #     'dimension': 3,
    #     'sizes': list(labels.shape),
    #     'encoding': 'gzip'
    # }
    # nrrd.write(str(outdir / 'crystal_filtered.nrrd'), labels, header)
    # plot_label_snapshot(labels)
    # plt.savefig(outdir / f'crystal_filtered.png', dpi=300, bbox_inches='tight', pad_inches=0)