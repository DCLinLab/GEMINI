import numpy as np
from skimage.filters import gaussian
from tqdm import tqdm
import os
import functools

def hide_output(func):
    """
    Decorator that suppresses all output (stdout and stderr) at the file-descriptor level,
    capturing output even from subprocesses or worker processes.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Open the null device for writing (discard all writes)
        null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # 2. Save original stdout (1) and stderr (2) file descriptors
        save_fds = [os.dup(1), os.dup(2)]
        try:
            # 3. Redirect stdout and stderr fds to the null device
            os.dup2(null_fds[0], 1)  # stdout → devnull
            os.dup2(null_fds[1], 2)  # stderr → devnull
            # 4. Execute the target function
            return func(*args, **kwargs)
        finally:
            # 5. Restore original stdout and stderr
            os.dup2(save_fds[0], 1)
            os.dup2(save_fds[1], 2)
            # 6. Close all file descriptors we opened or duplicated
            for fd in null_fds + save_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
    return wrapper


def get_canvas_range(positions, tile_width, tile_height):
    positions = np.asarray(positions)
    min_x = np.min(positions[:, 1])
    min_y = np.min(positions[:, 0])
    max_x = np.max(positions[:, 1]) + tile_width
    max_y = np.max(positions[:, 0]) + tile_height
    return min_x, min_y, max_x, max_y


def snr(img, sigma=1):
    # Apply Gaussian blur to estimate the signal
    smoothed = gaussian(img, sigma, preserve_range=True)

    # Estimate noise
    noise = img - smoothed

    # Compute signal and noise power
    signal_power = np.mean(smoothed ** 2)
    noise_power = np.mean(noise ** 2)

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def stitch_tiles(tiles, positions, canvas_range=None, remix=False, background=0):
    """
    Stitch image tiles into a single image with blending in overlapping regions.

    Parameters:
    - tiles: List of PIL.Image objects.
    - positions: List of (x, y) tuples indicating the top-left position of each tile.

    """
    # Convert images to numpy arrays
    tiles = np.asarray(tiles)
    tile_depth, tile_height, tile_width = tiles[0].shape

    # Determine the size of the final stitched image
    if canvas_range is None:
        canvas_range = get_canvas_range(positions, tile_width, tile_height)
    min_x, min_y, max_x, max_y = canvas_range

    # Initialize arrays for accumulating pixel values and counts

    if remix:
        stitched_image = np.ones((tile_depth, max_y - min_y, max_x - min_x), dtype=np.float32) * background
        count = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)

        # Accumulate pixel values and counts
        for tile_array, pos in tqdm(list(zip(tiles, positions)), 'Stitching'):
            y, x = pos[0], pos[1]
            y -= min_y
            x -= min_x
            stitched_image[:, y:y+tile_height, x:x+tile_width] += tile_array
            count[y:y+tile_height, x:x+tile_width] += 1

        # Avoid division by zero
        count[count == 0] = 1

        # Compute the average in overlapping regions
        stitched_image /= count

        # Convert back to uint8
        stitched_image = np.clip(stitched_image, 0, tiles.max()).astype(tiles.dtype)

    else:
        stitched_image = np.ones((tile_depth, max_y - min_y, max_x - min_x), dtype=tiles[0].dtype) * background
        for tile_array, pos in tqdm(list(zip(tiles, positions)), 'Stitching'):
            y, x = pos[0], pos[1]
            y -= min_y
            x -= min_x
            stitched_image[:, y:y + tile_height, x:x + tile_width] = tile_array

    # Convert numpy array back to PIL.Image
    return stitched_image


def safe_crop2d(img, ct, dim):
    # Compute crop boundaries so that the crop is centered at (center_y, center_x)
    crop_min_y = round(np.floor(ct[0] - dim / 2))
    crop_min_x = round(np.floor(ct[1] - dim / 2))
    crop_max_y = crop_min_y + dim
    crop_max_x = crop_min_x + dim

    # Initialize the crop array with zeros
    cropped = np.zeros((dim, dim), dtype=img.dtype)

    # Calculate the overlap between the crop and the image
    image_min_y = max(0, crop_min_y)
    image_max_y = min(img.shape[0], crop_max_y)
    image_min_x = max(0, crop_min_x)
    image_max_x = min(img.shape[1], crop_max_x)

    # Calculate corresponding indices in the crop array
    crop_start_y = max(0, -crop_min_y)
    crop_end_y = crop_start_y + (image_max_y - image_min_y)
    crop_start_x = max(0, -crop_min_x)
    crop_end_x = crop_start_x + (image_max_x - image_min_x)

    # Copy the overlapping part from the image to the crop
    if image_min_y < image_max_y and image_min_x < image_max_x:
        cropped[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = img[image_min_y:image_max_y, image_min_x:image_max_x]
    return cropped

