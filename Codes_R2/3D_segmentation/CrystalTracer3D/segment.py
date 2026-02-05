import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import reconstruction
from skimage.filters import threshold_triangle, threshold_otsu, difference_of_gaussians
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import remove_small_objects
from skimage.exposure import adjust_gamma
import pyclesperanto_prototype as cle
from .util import hide_output
from skimage.transform import rescale, resize
from skimage.morphology import binary_erosion
from skimage.measure import label
from skimage.segmentation import expand_labels
from skimage.exposure import match_histograms
from .gwdt import gwdt
from scipy.ndimage import generate_binary_structure
from tqdm.contrib.concurrent import process_map
from skimage.morphology import binary_opening
from skimage.morphology.footprints import disk
from skimage.measure import regionprops
from skimage.morphology.footprints import ball


def label_pruning(labels, radius: int):
    labels = np.asarray(labels)
    regions = regionprops(labels)
    new_labels = np.zeros_like(labels)
    for i in regions:
        img = i.image.max(axis=0)
        img = binary_opening(img, disk(radius), mode='min')
        temp = np.stack([img] * i.image.shape[0]) & i.image

        img = i.image.max(axis=1)
        img = binary_opening(img, disk(radius), mode='min')
        img = np.stack([img] * i.image.shape[1], axis=1) & temp

        new_labels[i.slice] += (img * i.label).astype('uint32')
    return new_labels


def stack_hist_matching(img, slice=None):
    match = []
    slice = slice or img.shape[0] // 2
    ref = img[slice]
    for i in img:
        match.append(match_histograms(i, ref))
    return np.stack(match)


def bgrm(img):
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.min()
    thr = np.percentile(img, 90)
    seed = seed.clip(None, thr)
    mask = img
    rec = reconstruction(seed, mask, method='dilation')
    return img - rec


@hide_output
def voronoi(img, spot_sigma, outline_sigma):
    return np.array(cle.voronoi_otsu_labeling(img, spot_sigma=spot_sigma, outline_sigma=outline_sigma))


def segment_soma(img: np.ndarray, sigma=3, spot_sigma=15, radius=3, zspacing=1.):
    s = img.shape
    img = gaussian(img, sigma)
    img = process_map(bgrm, img)
    img = rescale(np.array(img), (zspacing, 1, 1))
    labels = voronoi(img, spot_sigma, 1)
    # labels = label_pruning(labels, radius)
    labels = resize(labels, s, order=0, preserve_range=True, clip=False).astype(labels.dtype)
    return labels


def segment_filled_crystal_by_plane(img, gamma=0.5, sigma=3, spot_sigma=1, expand_dist=3, spacing=(4, 1, 1)):
    processed = []
    for i in img:
        # smooth and enhance weak ones
        i = gaussian(adjust_gamma(i, gamma), sigma)
        # fill holes
        seed = np.copy(i)
        seed[1:-1, 1:-1] = i.max()
        mask = i
        i = reconstruction(seed, mask, method='erosion')
        # bg removal
        dog = difference_of_gaussians(i, 0, sigma*3).clip(0)
        structure = generate_binary_structure(2, 10)
        dt = gwdt(dog, structure)
        processed.append(dt)
    processed = np.stack(processed)
    labels = voronoi(processed, spot_sigma, 0)
    labels = expand_labels(np.array(labels), expand_dist, spacing)
    return labels


@hide_output
def segment_filled_crystal(img: np.ndarray, dog_sigma=1., gamma=.5, dog=True,
                           spot_sigma=1., outline_sigma=1., zspacing=1., footprint_radius=1):
    s = img.shape
    img = rescale(img, (zspacing, 1, 1))
    if dog:
        img = difference_of_gaussians(img, dog_sigma, 10).clip(0)
    img = adjust_gamma(img, gamma)
    labels = cle.voronoi_otsu_labeling(img, spot_sigma=spot_sigma, outline_sigma=outline_sigma)

    if footprint_radius > 0:
        labels = labels > 0
        labels = binary_erosion(labels, footprint=ball(footprint_radius))
        labels, num = label(labels, connectivity=1, return_num=True)
        labels = expand_labels(labels, distance=footprint_radius)

    labels = resize(labels, s, order=0, preserve_range=True, clip=False).astype(labels.dtype)

    lookup = np.arange(labels.max()) + 1
    lookup = [0] + list(lookup)
    unq = list(np.unique([0] + list(np.unique(labels))))
    lookup = np.array([unq.index(i) if i in unq else 0 for i in lookup])
    labels = np.take(lookup, labels)

    return labels


def segment_vessel(img: np.ndarray, sigma=1, min_size=1000, thr=(-3, 3)):
    img = np.asarray(img)
    img = (img - img.mean()) / img.std()
    img = img.clip(*thr)
    img = difference_of_gaussians(img, (1, sigma, sigma)).clip(0)
    thr = threshold_otsu(img)
    seg = img > thr
    seg = remove_small_objects(seg, min_size, 3)
    return seg


def segment_crystal_whole(img: np.ndarray, res, sigma=(0, 1, 1), min_volume=.5, min_sphericity=.5):
    """
    Segment crystal regions from a 3D image based on intensity, morphological features,
    and surface-area-to-volume ratio criteria.

    This function applies a series of image processing steps to identify and segment
    crystal structures in a volumetric image. The process includes:
      1. Gaussian smoothing of the input image.
      2. Background removal using triangle threshold.
      3. Detection of holes in each 2D slice via 2D morphological reconstruction (erosion).
      4. Labeling of connected components within the hole mask.
      5. Computation of component sizes and their sphericity and filtering out regions not satisfying the threshold.
      6. Relabeling of the filtered mask to yield the final segmented regions.

    Parameters:
        img (np.ndarray): Input 3D image array to be segmented.
        res: z, y, x resolution in um.
        sigma (tuple, optional): Standard deviations for Gaussian smoothing along
            each axis (z, y, x). Default is (0, 2, 2), implying no smoothing along z.
        min_volume (float, optional): Minimum volume for a region to be considered.
        min_sphericity (float, optional): Minimum sphericity for a region to be considered.
    Returns:
        np.ndarray: A 3D labeled array where each valid crystal region is assigned a unique label.

    Notes:
        - The function relies on external libraries such as scipy.ndimage (imported as ndi),
          skimage.filters.gaussian, skimage.morphology.reconstruction, and tqdm for progress display.
        - The algorithm assumes that crystal structures appear as holes in the processed image,
          which are further validated based on size and morphology.
    """

    # smoothing
    smoothed = gaussian(img, sigma)

    smoothed = smoothed.clip(threshold_triangle(smoothed))

    # find holes in 2D
    holes = []
    # get holes
    border_width = 1
    for i in smoothed:
        # hole
        seed = np.copy(i)
        seed[border_width:-border_width, border_width:-border_width] = seed.max()
        rec = reconstruction(seed, i, method='erosion')
        holes.append(rec - i)
    holes = np.array(holes)

    # Compute sizes of labeled components
    labels, nfeatures = ndi.label(holes)
    V = []
    A = []
    bbox = ndi.find_objects(labels)
    for i in range(1, nfeatures + 1):
        # get sub region
        region = np.pad(labels[bbox[i - 1]] == i, pad_width=1, mode='constant', constant_values=0)
        # fill hole
        seed = np.copy(region)
        seed[1:-1, 1:-1] = True
        region = reconstruction(seed, region, method='erosion')

        volume = np.sum(region) * res[0] * res[1] * res[2]
        V.append(volume)

        if region.shape[0] < 2 or region.shape[1] < 2 or region.shape[2] < 2:
            A.append(volume)
        else:
            vert, face = marching_cubes(region, 0.5, spacing=(res[0], res[1], res[2]))[:2]
            surface_area = mesh_surface_area(vert, face)
            A.append(surface_area)
    A = np.array(A)
    V = np.array(V)
    sph = (36 * np.pi * V ** 2) ** (1 / 3) / A

    # Create a mask of valid regions
    filtered_mask = np.isin(labels, np.where((V >= min_volume) & (sph >= min_sphericity))[0] + 1)

    # Re-label the filtered mask
    filtered_labels, _ = ndi.label(filtered_mask)
    return filtered_labels


