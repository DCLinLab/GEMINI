from skimage.measure import regionprops
from functools import cached_property
import numpy as np


class SegmentationAnalyzer:
    def __init__(self, labels, res):
        self._labels = labels
        self._regions = regionprops(labels, spacing=res)

    @cached_property
    def volume(self):
        area = [i.area for i in self._regions]
        return np.array(area)

    def extract_avg_intensity(self, intensity):
        out = []
        for r in self._regions:
            c = r.coords
            avg = intensity[c[:, 0], c[:, 1], c[:, 2]].mean()
            out.append(avg)
        return np.array(out)
