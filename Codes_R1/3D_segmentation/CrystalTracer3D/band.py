import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import map_coordinates, shift
from skimage.transform import resize
from functools import cached_property
from skimage.filters import gaussian
from scipy.signal import find_peaks, peak_prominences
from .io import CrystalReader
from .util import safe_crop2d
from skimage.transform import hough_line, hough_line_peaks
from skimage.segmentation import inverse_gaussian_gradient
from skimage.feature import canny
from sklearn.neighbors import KDTree


def merge_lines(scores, angles, dists, dmin):
    out_scores = []
    out_angles = []
    for i, a in enumerate(angles):
        s = 0
        k = 0.
        for j, b in enumerate(angles):
            d = min(abs(a - b), abs(a + b))
            if d < dmin:
                w = scores[j] / (1 + abs(abs(dists[j]) - abs(dists[i])))
                s += w
                if d == abs(a - b):
                    k += w * b
                else:
                    k -= w * b
        k /= s
        out_angles.append(k)
        out_scores.append(s)
    return out_scores, out_angles



class SegmentationAnalyzer:
    def __init__(self, labels: np.ndarray, raw: np.ndarray, seg_chan, res,
                 max_crop=30, magnify=10, enlarge=10):
        """
        Initialize the analyzer with a 3D labeled image and precompute region properties.

        Processes the input labeled image to compute 3D region properties using regionprops. For each region,
        the central 2D section is extracted by computing the relative z-index from the region's bounding box and
        its centroid. The extracted 2D sections are then analyzed further.

        Parameters:
            labels (numpy.ndarray): A 3D labeled image from which to compute region metrics.
            image (numpy.ndarray): A 4D image from which to compute region properties.
        """
        self._props_3d = regionprops(labels)
        self._props_section = []
        self._seg_chan = seg_chan

        for region in self._props_3d:
            rel_z = round(region.centroid[0]) - region.slice[0].start
            self._props_section.extend(regionprops(region.image[rel_z] * region.label))

        kd = KDTree(self.centers * res)
        d, _ = kd.query(self.centers, k=2)
        d = d[:, 1]

        ans = []
        for i, r in enumerate(self._props_section):
            ct = self.centers[i]
            level = round(ct[0])
            y1, x1, y2, x2 = r.bbox
            dim = min(max(x2 - x1, y2 - y1) + enlarge, max_crop)
            if d[i] < dim * res[-1] / 2:
                continue
            dim2 = dim * magnify

            # gradient image (smoothed and inverted from the raw image)
            gd = safe_crop2d(raw[level], ct[1:], dim)
            gd = resize(gd, (dim2, dim2))
            gd = inverse_gaussian_gradient(gd)

            # canny
            cont = canny(gd, sigma=3, use_quantiles=True, low_threshold=0.1, high_threshold=0.9)
            counts, angles, dists = hough_line_peaks(
                *hough_line(cont, np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)))
            if len(counts) == 0:
                continue
            counts, angles = merge_lines(counts, angles, dists, np.pi / 90)
            a = angles[np.argmax(counts)]
            v = np.array([np.sin(a), np.cos(a)])
            ans.append(v)
        self.profile_directions = ans

    @cached_property
    def centers(self):
        """
        Retrieve the centroids of all segmented regions.

        Returns:
            numpy.ndarray: A read-only array of shape (N, 3), where N is the number of regions, containing the
                           (z, y, x) coordinates of each region's centroid.
        """
        ans = np.array([(r3.centroid[0], r3.slice[1].start + r2.centroid[0],
                         r3.slice[2].start + r2.centroid[1]) for r3, r2 in zip(self._props_3d, self._props_section)])
        ans.flags.writeable = False
        return ans

    @cached_property
    def mask_radius(self):
        """
        Compute the radius of the mask along the profile direction.

        :return:
        """
        ans = []

        def march(c, d, mask):
            dist = 0
            value = 1
            while value > 0.5:
                dist += 1
                point = c + d * dist
                # Check boundaries: if outside mask bounds, break
                if not (0 <= point[0] < mask.shape[0] and 0 <= point[1] < mask.shape[1]):
                    break
                # Use interpolation to get the mask value at the continuous coordinate
                value = map_coordinates(mask.astype(float), [[point[0]], [point[1]]], order=0)[0]
            return dist

        for d, r in zip(self.profile_directions, self._props_section):
            mask = r.image_convex
            ans.append(max(march(r.centroid, d, mask), march(r.centroid, -d, mask)))

        ans = np.array(ans)
        ans.flags.writeable = False
        return ans

    @cached_property
    def section_areas(self):
        """
        Get the filled area of the 2D central section for each segmented region.

        Returns:
            numpy.ndarray: A read-only array containing the filled area measurements for each region's central section.
        """
        ans = np.array([region.area_filled for region in self._props_section])
        ans.flags.writeable = False
        return ans


def profile_snr(prof: np.ndarray):
    return prof.max(axis=-1) / (prof.std(axis=-1) + 1e-9)


def profile_symmetry(prof: np.ndarray):
    cor = [np.corrcoef(p, p[::-1])[0, 1] for p in prof.reshape(-1, prof.shape[-1])]
    return np.array(cor).reshape(prof.shape[:-1])


def get_shift(prof, pct=0.8):
    ct = len(prof) // 2
    prof1 = prof[:ct]
    prof2 = prof[ct:]
    p1 = np.where(prof1 >= pct * prof1.max())[0][-1]
    p2 = np.where(prof2 >= pct * prof2.max())[0][0] + ct
    return (len(prof) - 1 - p1 - p2) / 2


def get_cutoff(half):
    x0 = np.argmax(half)
    x1 = x0 + np.argmin(half[x0:])
    dx = x1 - x0
    if dx == 0:
        return x0
    y0, y1 = half[x0], half[x1]
    dy = y1 - y0
    distances = []
    denominator = np.sqrt(dy ** 2 + dx ** 2)
    for i in range(x0, x1 + 1):
        numerator = abs(dy * i - dx * (half[i] - y0) + x1 * y0 - y1 * x0)
        distances.append(numerator / denominator)
    return x0 + np.argmax(distances)


class ProfileAnalyzer:
    """
    Analyze intensity profiles from an image along specified directions and compute summary metrics.

    This class extracts intensity profiles from a multi-channel image by sampling along lines defined by
    given center coordinates and direction vectors. For each region, the profile is generated by sampling
    along a line of a specified length and width. The resulting profiles are then used to compute metrics
    such as symmetry, signal-to-noise ratio (SNR), and maximum peak prominences.
    """

    def __init__(self, reader: CrystalReader, channels, centers, directions,
                 max_diam_um=15, width=5, num_samples=100, cutting=True,
                 min_snr=2, min_symmetry=0.8, min_cutoff_um=5, sigma_um=0.2, filter=True):
        """
        Initialize the ProfileAnalyzer with the image data and sampling parameters.

        Parameters:
            image (numpy.ndarray): A multi-channel image array from which profiles will be extracted.
                                   Expected dimensions include a channel axis.
            centers (array-like): Coordinates for the centers of the profiles, where each center is given
                                  as (level, y, x), with 'level' indicating the slice to sample.
            directions (array-like): Vectors representing the principal directions along which profiles are sampled.
            max_diam_um (int, optional): The half-length of the profile line to sample on each side of the center. Default is 50.
            width (int, optional): The number of parallel lines (width of the profile) to sample for averaging. Default is 5.
            num_samples (int, optional): The number of sampling points along each profile line. Default is 100.
        """
        res = reader.resolution[-1]
        min_cutoff = round(min_cutoff_um / res)
        sigma = sigma_um / res
        self.max_diam = d = round(max_diam_um / res)
        self.width = width
        self.channels = channels
        self.num_samples = num_samples
        self.centers = []
        self.directions = []
        self.profiles = []
        self.unit_dist = []

        nchan = len(channels)
        for center, direction in zip(centers, directions):
            perp = np.array([-direction[1], direction[0]])  # Get perpendicular vector
            prof = [[] for i in range(nchan)]

            # Generate profile lines with width
            ct = np.array([d / 2, d / 2])
            for w in np.linspace(-width // 2, width // 2, width):
                # Define start and end points of the profile
                start = ct - d / 2 * direction + w * perp
                end = ct + d / 2 * direction + w * perp

                # Generate sample coordinates along the line
                y_samples = np.linspace(start[0], end[0], num_samples)
                x_samples = np.linspace(start[1], end[1], num_samples)
                coords = np.vstack([y_samples, x_samples])
                # Sample intensity values using interpolation
                for ch, p in zip(channels, prof):
                    img = reader.crop(ch, round(center[0]), center[1:], d, d)
                    p.append(map_coordinates(img, coords, order=1))    # Bilinear interpolation of one column

            prof = np.array(prof).mean(axis=1)  # averaging across width
            prof = gaussian(prof, sigma, channel_axis=0)      # smoothing

            # cutoff
            if cutting:
                ct = num_samples // 2
                cutoff = max(*[1.2 * get_cutoff(np.min([ch[:ct][::-1], ch[ct:]], axis=0)) for ch in prof])
                cutoff = min(ct, round(cutoff))
                if cutoff < min_cutoff:
                    if filter:
                        continue        # discard when cutoff is too small
                    else:
                        cutoff = min_cutoff
                prof = prof[:, ct - cutoff:ct + cutoff]
                prof = resize(prof, (nchan, num_samples), order=1, preserve_range=True, mode='edge')

                self.unit_dist.append(cutoff * 2 / num_samples * res)
            else:
                self.unit_dist.append(res)

            self.profiles.append(prof)
            self.centers.append(center)
            self.directions.append(direction)

        # quality control the profiles
        self.profiles = np.array(self.profiles)
        self.centers = np.array(self.centers)
        self.directions = np.array(self.directions)

        if filter:
            snr = profile_snr(self.profiles).min(axis=1)  # min snr for each profile
            sym = profile_symmetry(self.profiles).min(axis=1)  # min symmetry for each profile
            choose = (snr >= min_snr) & (sym >= min_symmetry)
            self.profiles = self.profiles[choose]
            self.centers = self.centers[choose]
            self.directions = self.directions[choose]
            print(f'Retained: {len(self.profiles)}, Dropped: {len(centers) - len(self.profiles)}')

    @cached_property
    def aligned_profiles(self) -> np.ndarray:
        ans = []
        for k, prof in enumerate(self.profiles):
            ans.append([shift(i, get_shift(i), mode='nearest') for i in prof])
        ans = np.array(ans)
        return ans

    @cached_property
    def scaled_half_profiles(self):
        ct = self.num_samples // 2
        ans = []
        for prof in self.aligned_profiles:
            a = prof[:, :ct][:, ::-1]
            a = a[:max(*[get_cutoff(ch) for ch in a])]
            a = resize(a, (prof.shape[0], self.num_samples), order=1, preserve_range=True, mode='edge')
            a = a / a.max(axis=1)[:, None]
            ans.append(a)
            b = prof[:, ct:]
            b = b[:max(*[get_cutoff(ch) for ch in b])]
            b = resize(b, (prof.shape[0], self.num_samples), order=1, preserve_range=True, mode='edge')
            b = b / b.max(axis=1)[:, None]
            ans.append(b)
        ans = np.array(ans)
        return ans
