import yaml
import fire
import nrrd
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.exposure import is_low_contrast


from CrystalTracer3D.intensity import *
from CrystalTracer3D.io import *
from CrystalTracer3D.segment import *
from CrystalTracer3D.util import *
from CrystalTracer3D.plot import *


def _tile_take(args):
    lut, path = args
    indices, _ = nrrd.read(str(path / 'crystal.nrrd'))
    return np.take(lut, indices)


def _import_tile(args):
    path, ves_chan = args
    reader = CrystalReader(path)
    return reader.read(reader.find_channel(ves_chan))


def _import_pos(path):
    return CrystalReader(path).coord()


def _take(args):
    lut, path = args
    lut = [-1] + list(lut)       # label starts from 1, bg will be marked as -1
    indices, _ = nrrd.read(str(path / 'crystal.nrrd'))
    tile = np.take(lut, indices)
    return tile


def _segment_crystal_tile(args):
    in_img, out_base, seg_chan, int_chans, gamma, sigma, spot_sigma, expand_dist = args
    outdir = out_base / in_img.stem
    reader = CrystalReader(in_img)
    chan = reader.find_channel(seg_chan)
    img = reader.read(chan)
    if is_low_contrast(img.max(axis=0)):
        print(f"Skipping {in_img} for low contrast")
        return
    img2 = stack_hist_matching(img)
    spacing = np.array(reader.resolution) / reader.resolution[-1]
    labels = segment_filled_crystal_by_plane(img2, gamma, sigma, spot_sigma, expand_dist, spacing)
    outdir.mkdir(parents=True, exist_ok=True)
    header = {
        'dimension': 3,
        'sizes': list(labels.shape),
        'encoding': 'gzip'
    }
    nrrd.write(str(outdir / 'crystal.nrrd'), labels, header)
    imsave(outdir / 'raw_mip.png', img_as_ubyte(img).max(axis=0))
    plot_label_snapshot(labels)
    plt.savefig(outdir / 'crystal.png', dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.close()
    a = SegmentationAnalyzer(labels, reader.resolution)
    table = {}
    for ch in int_chans:
        c = reader.find_channel(ch)
        im = reader.read(c)
        # im = stack_hist_matching(im)
        table[ch] = a.extract_avg_intensity(im)
    table['volume'] = a.volume
    pd.DataFrame(table).to_csv(outdir / 'intensity.csv', index=False)


def _paint_tile(args):
    """
    args: tuple of (vals, path)
      - vals: 1D array of length N, with intensity values mapped to [0,255]
      - path: Path to directory containing 'crystal.nrrd'

    Returns a uint8 numpy array same shape as segmentation,
    where each labeled region k (1…N) is painted with vals[k-1].
    """
    vals, path = args
    seg, _ = nrrd.read(str(path / 'crystal.nrrd'))  # segmentation labels
    tile = np.zeros_like(seg, dtype=np.uint8)
    for k, v in enumerate(vals):
        tile[seg == (k + 1)] = int(round(v))
    return tile


class VesselCrystalCLI:
    def __init__(self, config='config.yml'):
        if not Path(config).is_file():
            print('Configuration file not found, use default configuration.')
            config = Path(__file__).parent / 'config.yml'
        with open(config, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def segment_crystal_by_tile(
        self,
        in_path: str,
        out_root: str,
        seg_chan: str = None,
        int_chans: list = None,
        gamma: float = None,
        sigma: float = None,
        spot_sigma: float = None,
        expand_dist: float = None,
        workers: int = 4
    ):
        # 覆盖默认设置
        cfg = self.cfg
        in_path = Path(in_path)
        out_base = Path(out_root)
        seg_chan = seg_chan or cfg['crystal']['seg_chan']
        int_chans = int_chans or cfg['crystal']['int_chan']
        gamma = gamma or cfg['crystal']['gamma']
        sigma = sigma or cfg['crystal']['sigma']
        spot_sigma = spot_sigma or cfg['crystal']['spot_sigma']
        expand_dist = expand_dist or cfg['crystal']['expand_dist']

        if in_path.is_dir():
            imgs = sorted(in_path.glob('*.czi'))
        else:
            # if the input path is an image, the tiles would be itself plus the files with
            # the same naming convention
            imgs = [in_path] + sorted(in_path.parent.glob(f'{in_path.stem}(*).czi'))
        arg_list = [(i, out_base, seg_chan, int_chans, gamma, sigma, spot_sigma, expand_dist) for i in imgs]
        process_map(_segment_crystal_tile, arg_list, max_workers=workers)

    def ratiometrics(self,
                     segmentation_root: str = None,
                     raw_root: str = None,
                     out_name: str = 'ratiometrics',
                     volume_thr: float = None,
                     mip: bool = False,
                     workers: int = 4,
                     colorbar_range = (1e-2, 10)):
        """Assemble heatmap from segmented tiles"""
        cfg = self.cfg
        resdir = Path(segmentation_root)
        rawdir = Path(raw_root)
        c1, c2 = cfg['crystal']['int_chan']
        paths = list(resdir.iterdir())

        all_imgs = [i for i in rawdir.glob('*.czi') if i.stem.endswith(')')]
        all_pos = [CrystalReader(i).coord() for i in all_imgs]
        bbox = CrystalReader(all_imgs[0]).czi.get_scene_bounding_box()
        cr = get_canvas_range(all_pos, bbox.w, bbox.h)

        lut, pos = [], []
        for p in tqdm(paths):
            reader = CrystalReader(rawdir / p.with_suffix('.czi').name)
            pos.append(reader.coord())
            table = pd.read_csv(p / 'intensity.csv')
            ratio = table[c1] / table[c2]
            # ratio[table['volume'] < vol_thr] = 0
            lut.append(np.array(ratio))

        # normalization
        # all_vals = np.hstack(lut)
        # low, high = (np.percentile(all_vals, cfg['assemble'][f'{pct}_pct'])
        #              for pct in ('low', 'high'))
        # clipped = [((v.clip(low, high) - low) / (high - low) * 255).astype(np.int16) for v in lut]

        tiles = process_map(_take, list(zip(lut, paths)), max_workers=workers)

        if mip:
            tiles = [t.max(axis=0, keepdims=True) for t in tiles]

        final = stitch_tiles(tiles, pos, cr, background=-1)

        if mip:
            final = final.squeeze()

        header = {
            'dimension': len(final.shape),
            'sizes': list(final.shape),
            'encoding': 'gzip'
        }
        nrrd.write(str(resdir.parent / f'{out_name}.nrrd'), final, header)
        plot_heatmap_mip(final, colorbar_range=colorbar_range)
        plt.savefig(resdir.parent / f'{out_name}.png', dpi=300)
        plt.close()

    def segment_vessel(self,
                       in_path: str,
                       out_root: str,
                       ves_chan: str = None,
                       seg_scale: float = None,
                       thr_range: float = None,
                       workers: int = 4):
        """
        Stitch vessel tiles, segment vessels, save NRRD + MIP.
        Shares defaults from config but can be overridden via CLI.
        """
        cfg = self.cfg
        in_path = Path(in_path)
        outdir = Path(out_root)
        seg_scale = seg_scale or cfg['vessel']['seg_scale']
        ves_chan = ves_chan or self.cfg['vessel']['chan']
        thr = thr_range or cfg['vessel']['thr']

        if in_path.is_dir():
            imgs = sorted(in_path.glob('*.czi'))
        else:
            # if the input path is an image, the tiles would be itself plus the files with
            # the same naming convention
            imgs = [in_path] + sorted(in_path.parent.glob(f'{in_path.stem}(*).czi'))

        arglist = [(i, ves_chan) for i in imgs]
        tiles = process_map(_import_tile, arglist, max_workers=workers)
        pos = process_map(_import_pos, imgs, max_workers=workers)
        whole = stitch_tiles(tiles, pos)

        scaled = rescale(whole, (1, seg_scale, seg_scale), anti_aliasing=True, preserve_range=True)
        # scaled = stack_hist_matching(scaled, 1)
        seg = segment_vessel(scaled, thr=thr)
        seg = resize(seg, whole.shape, preserve_range=True, anti_aliasing=False)

        header = {
           'type': 'unsigned char',
           'dimension': 3,
           'sizes': list(seg.shape),
           'encoding': 'gzip'
       }
        nrrd.write(str(outdir / 'ves_seg.nrrd'), img_as_ubyte(seg),header)
        imsave(outdir / 'ves_seg_mip.png', img_as_ubyte(seg.max(axis=0)))
        print('Vessel segmentation complete.')

    def assemble_crystal(
            self,
            segmentation_root: str = None,
            raw_root: str = None,
            volume_thr: float = None,
            workers: int = 4
    ):
        """
        Merge segmented crystal tiles into a final labeled volume plus snapshot.
        与原始脚本功能一致。
        """
        cfg = self.cfg
        resdir = Path(segmentation_root or cfg['paths']['crystal'])
        rawdir = Path(raw_root or cfg['paths']['image'])
        vol_thr = volume_thr or cfg['assemble']['vol_thr']

        # 获取所有tile的位置与拼接画布大小
        all_imgs = [i for i in rawdir.glob('*.czi') if i.stem.endswith(')')]
        all_pos = [CrystalReader(i).coord() for i in all_imgs]
        reader0 = CrystalReader(all_imgs[0])
        bbox = reader0.czi.get_scene_bounding_box()
        canvas = get_canvas_range(all_pos, bbox.w, bbox.h)

        paths = sorted(resdir.iterdir())
        lut_list = []
        pos_list = []
        total = 0

        for p in paths:
            raw_img = rawdir / p.with_suffix('.czi').name
            reader = CrystalReader(raw_img)
            pos_list.append(reader.coord())

            table = pd.read_csv(p / 'intensity.csv')
            lookup = np.arange(len(table)) + 1
            lookup[table['volume'] < vol_thr] = 0
            lookup = [0] + list(lookup)
            unq = list(np.unique(lookup))
            lookup = np.array([unq.index(i) for i in lookup])
            lookup[lookup > 0] += total
            total = lookup.max()
            lut_list.append(lookup)

        tiles = process_map(_tile_take, list(zip(lut_list, paths)), max_workers=workers)
        final = stitch_tiles(tiles, pos_list, canvas)

        header = {
            'dimension': 3,
            'sizes': list(final.shape),
            'encoding': 'gzip'
        }
        out_nrrd = resdir.parent / 'crystal_whole.nrrd'
        nrrd.write(str(out_nrrd), final, header)
        plot_label_snapshot(final)
        plt.savefig(resdir.parent / 'crystal_whole.png', dpi=300,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f'Assembled volume saved to {out_nrrd}')

if __name__ == '__main__':
    fire.Fire(VesselCrystalCLI)