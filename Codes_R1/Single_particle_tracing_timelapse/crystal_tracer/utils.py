from pathlib import Path
from aicspylibczi import CziFile
import sys
import os


def get_czi_shape(path: Path) -> tuple[int, int, int, int, float]:
    czi = CziFile(path)
    dim = czi.dims
    size = czi.size
    try:
        interval = float(czi.meta.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').
                        find('SubDimensionSetups').find('TimeSeriesSetup').find('Duration').find('Cycles').text)
    except:
        interval = 100
    t = size[dim.index('T')]
    c = size[dim.index('C')]
    y = size[dim.index('Y')]
    x = size[dim.index('X')]
    return t, c, y, x, interval


def load_czi_slice(path: Path, channel: int, page: int):
    czi = CziFile(path)
    img = czi.read_image(T=page, C=channel)[0].squeeze()
    return img


class HidePrint:
    def __init__(self):
        self.origin = None

    def __enter__(self):
        sys.stdout.close()
        sys.stdout = self.origin

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.origin = sys.stdout
        sys.stdout = open(os.devnull, 'w')

