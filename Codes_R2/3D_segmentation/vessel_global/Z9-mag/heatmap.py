import yaml
import nrrd

from CrystalTracer3D.io import *
from CrystalTracer3D.plot import *


with open('config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


if __name__ == '__main__':
    in_img = r"Y:\All Data\Lab Members\Jiaxi Lu\02_Processed Data\20250505_ReviewData\3_invivo_NFkB_recording\3_3_SpatialRecording\WholeSlice_Nude32IHC\z9highmag_Nude34_Z9_2025_05_15__02_30_42-Create Image Subset-48.czi"
    outdir = Path(r'D:\Zuohan\Z9-mag')

    c1, c2 = cfg['crystal']['int_chan']

    reader = CrystalReader(in_img)
    table = pd.read_csv(outdir / 'intensity.csv')
    ratio = table[c1] / table[c2]
    # ratio[table['volume'] < cfg["assemble"]["vol_thr"]] = 0

    # normalization
    m1 = np.percentile(ratio, cfg['assemble']['low_pct'])
    m2 = np.percentile(ratio, cfg['assemble']['high_pct'])
    # ratio = ratio.clip(m1, m2)
    # ratio = ((ratio.clip(m1, m2) - m1) / (m2 - m1) * 255).round()
    ratio = [-1] + list(ratio)
    indices, _ = nrrd.read(str(outdir / 'crystal.nrrd'))
    heatmap = np.take(ratio, indices)
    # assign the values onto each segmented tile

    # output
    header = {
        'dimension': 3,
        'sizes': list(heatmap.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(outdir / 'ratiometrics.nrrd'), heatmap, header)
    plot_heatmap_mip(heatmap)
    plt.savefig(outdir / 'example.png', dpi=300)
