from CrystalTracer3D.plot import *
from CrystalTracer3D.io import *
from CrystalTracer3D.band import *
import yaml
from tqdm.contrib.concurrent import process_map


with open('../vessel_density_local/config.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

def main(in_dir):
    reader = CrystalReader(Path(cfg['paths']['image']) / f'{in_dir.name}.czi')
    outdir = Path(cfg['paths']['profile']) / in_dir.name
    crystal_chans = [reader.find_channel(i) for i in cfg['profiling']['crystal_ch']]
    labels = np.load(in_dir / 'crystal.npy')

    raw = np.array([reader.read(i) for i in crystal_chans])
    a1 = SegmentationAnalyzer(labels, raw, crystal_chans, reader.resolution)
    a2 = ProfileAnalyzer(reader, crystal_chans, a1.centers / (1, labels[1], labels[2]), a1.profile_directions,
                         filter=cfg['profiling']['filter'], cutting=cfg['profiling']['cutting'])

    export_training_data(outdir / 'table', a2)
    plot_profiles(outdir / 'plots', a2, palette=cfg['profiling']['palette'])
    plot_measurement(outdir / 'crops', a2, reader, palette=cfg['profiling']['palette'])


if __name__ == '__main__':
    indir = Path(cfg['paths']['segmentation'])
    process_map(main, sorted(indir.iterdir()))
