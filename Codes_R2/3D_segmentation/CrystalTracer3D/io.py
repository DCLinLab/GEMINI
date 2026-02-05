import numpy as np
from pathlib import Path
from aicspylibczi import CziFile
import pandas as pd
from distutils.util import strtobool



class CrystalReader:
    def __init__(self, path):
        self.czi = CziFile(path)
        self._channel_maps = self._extract_channel_maps()

    def _extract_channel_maps(self):
        # Extract channel names from metadata
        channels = {}
        i = 0
        for channel in self.czi.meta.findall(".//Channels/Channel"):
            at = channel.attrib
            if 'IsActivated' in at and not strtobool(at['IsActivated']):
                continue
            name = at["Name"]
            if name in channels:
                continue
            if name:
                channels[name] = i
                i += 1
        return channels

    def find_channel(self, name):
        for k in self.channel_maps:
            if name in k:
                return self.channel_maps[k]

    @property
    def channel_maps(self):
        return self._channel_maps

    @property
    def nchan(self):
        return len(self._channel_maps)

    @property
    def resolution(self):
        out = {}
        for i in self.czi.meta.find('Metadata').find('Scaling').find('Items').findall('*'):
            key = i.attrib['Id']
            out[key] = float(i.find('Value').text) * 1e6
        return out['Z'], out['Y'], out['X']

    def assemble_tiles(self, channel, scale=1):
        """
        Assemble tiles across specified Z-slices into one downscaled 3D mosaic.

        Parameters:
          scale    : float, downscale factor (<1 to shrink, >1 to upscale)
          channel  : int, channel index to extract
        Returns:
          np.ndarray : 3D array of shape (Z, H_ds, W_ds)
        """
        dims = self.czi.get_dims_shape()[0]
        zdim = dims['Z'][1]
        img = np.stack([np.squeeze(self.czi.read_mosaic(scale_factor=scale, C=channel, Z=i)) for i in range(zdim)])
        return img

    def crop(self, channel: int, level: int, center, height: int, width: int):
        temp = self.czi.get_mosaic_bounding_box()
        center = np.asarray(center) * (temp.h, temp.w) + (temp.y, temp.x)
        # Compute crop boundaries so that the crop is centered at (center_y, center_x)
        crop_min_y = round(center[0] - height / 2)
        crop_min_x = round(center[1] - width / 2)
        crop_max_y = crop_min_y + height
        crop_max_x = crop_min_x + width

        # Calculate the overlap between the crop and the image
        image_min_y = max(temp.y, crop_min_y)
        image_max_y = min(temp.h + temp.y, crop_max_y)
        image_min_x = max(temp.x, crop_min_x)
        image_max_x = min(temp.w + temp.x, crop_max_x)
        crop_width = image_max_x - image_min_x
        crop_height = image_max_y - image_min_y

        # Calculate corresponding indices in the crop array
        crop_start_y = image_min_y - crop_min_y
        crop_end_y   = crop_start_y + crop_height
        crop_start_x = image_min_x - crop_min_x
        crop_end_x   = crop_start_x + crop_width

        region = [image_min_x, image_min_y, crop_width, crop_height]
        img = np.squeeze(self.czi.read_mosaic(region, C=channel, Z=level))
        cropped = np.zeros((height, width), dtype=img.dtype)
        cropped[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = img
        return cropped

    def read(self, channel):
        img, _ = self.czi.read_image(C=channel)
        return np.squeeze(img)

    def coord(self):
        try:
            bbox = self.czi.get_mosaic_bounding_box()
        except:
            bbox = self.czi.get_scene_bounding_box()
        return bbox.y, bbox.x


def export_segmentation_analysis(out_dir, analyzer):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'center_x': analyzer.centers[:, 2], 'center_y': analyzer.centers[:, 1],
                       'center_z': analyzer.centers[:, 0], 'profile_direction_x': analyzer.profile_directions[:, 1],
                       'profile_direction_y': analyzer.profile_directions[:, 0],
                       'section_areas': analyzer.section_areas, 'msk_radius': analyzer.mask_radius})
    df.to_csv(out_dir / 'segmentation_metrics.csv')
        
        
def export_training_data(out_dir, analyzer):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_values = np.arange(-analyzer.num_samples // 2, analyzer.num_samples // 2)
    with pd.ExcelWriter(out_dir / 'profiles.xlsx') as writer:
        for i, ch in enumerate(analyzer.scaled_half_profiles.transpose([1, 0, 2])):
            df = pd.DataFrame(ch, columns=x_values)
            df.to_excel(writer, sheet_name=f'CH{i}')
                
                
def export_profiles(out_dir, analyzer):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_values = np.arange(-analyzer.num_samples // 2, analyzer.num_samples // 2)
    for i, prof in enumerate(analyzer.profiles):
        df = pd.DataFrame({'distance': x_values * analyzer.unit_dist[i],
                           **{f'CH{j}': ch for j, ch in enumerate(prof)}})
        df.to_csv(out_dir / f'{i + 1}.csv', index=False)


