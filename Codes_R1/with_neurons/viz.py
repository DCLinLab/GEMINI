import nrrd
import yaml
from CrystalTracer3D.io import CrystalReader
from skimage.segmentation import expand_labels
import napari
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import rankdata


def show_scale_bar():
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'μm'
    viewer.scale_bar.length = 20
    viewer.scale_bar.font_size *= 1.5


def set_camera():
    viewer.camera.perspective = 30
    viewer.camera.zoom = 5
    viewer.camera.angles = (-1.5, 20, 85)


def set_bbox(layer):
    bbox = layer.bounding_box
    bbox.visible = True
    bbox.line_color = 'white'
    bbox.line_thickness = 2
    bbox.point_size = 0


if __name__ == '__main__':
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        repo = Path(cfg['data']['segmentation'])
        slice_range = cfg['data']['range']
    lab_n, _ = nrrd.read(str(repo / 'neuron.nrrd'))
    lab_c, _ = nrrd.read(str(repo / 'crystal.nrrd'))

    reader = CrystalReader(cfg['data']['path'])
    ch = reader.find_channel(cfg['data']['neuron'])
    img_n = reader.read(ch)[slice_range[0]:slice_range[1]]
    ch = reader.find_channel(cfg['data']['crystal'])
    img_c = reader.read(ch)[slice_range[0]:slice_range[1]]
    lab_n = expand_labels(lab_n, 1, spacing=reader.resolution)

    # raw images
    viewer = napari.Viewer(ndisplay=3)
    layers = viewer.add_image(np.stack([img_n, img_c]), channel_axis=0, name=['neuron', 'crystal'],
                              colormap=["green", "red"], blending=['translucent', "additive"], scale=reader.resolution)
    layers[0].contrast_limits = cfg['data']['neuron_contrast']
    layers[1].contrast_limits = cfg['data']['crystal_contrast']
    set_bbox(layers[0])
    set_bbox(layers[1])
    set_camera()
    show_scale_bar()

    viewer.screenshot(path=str(repo / "both_raw.png"), canvas_only=True)
    viewer.scale_bar.visible = False
    viewer.screenshot(path=str(repo / "both_raw_no_scale.png"), canvas_only=True)
    layers[0].visible = False
    viewer.screenshot(path=str(repo / "crystal_raw.png"), canvas_only=True)
    layers[1].visible = False
    layers[0].visible = True
    viewer.screenshot(path=str(repo / "neuron_raw.png"), canvas_only=True)
    viewer.close()

    # segmentation
    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_labels(lab_n, opacity=.8, scale=reader.resolution)
    layer.iso_gradient_mode = 'smooth'
    layer = viewer.add_labels(lab_c, opacity=.8, scale=reader.resolution)
    layer.iso_gradient_mode = 'smooth'
    layer = viewer.add_image(img_n, opacity=.7, blending='additive', scale=reader.resolution)
    layer.contrast_limits = cfg['data']['neuron_contrast']
    set_bbox(layer)
    set_camera()

    viewer.layers[1].visible = False
    viewer.screenshot(path=str(repo / "neuron_seg.png"), canvas_only=True)
    viewer.layers[1].visible = True
    viewer.layers[0].visible = False
    viewer.screenshot(path=str(repo / "crystal_seg.png"), canvas_only=True)
    viewer.close()

    neurons_df = pd.read_csv(repo / "neuron_prop.csv", index_col='label')
    pairs = pd.read_csv(repo / "pair.csv")
    remove = cfg['data']['drop_neuron']
    neurons_df['rank'] = rankdata(neurons_df.index, method='min')
    pairs = pairs[pairs['neuron'].isin(neurons_df[~neurons_df['rank'].isin(remove)].index)]
    pairs.to_csv(repo / "pair_filtered.csv", index=False)

    a1 = np.isin(lab_c, pairs['crystal'])
    a2 = np.isin(lab_n, pairs['neuron'])

    # paired
    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_image(img_n, opacity=.7, blending='additive', scale=reader.resolution)
    layer.contrast_limits = cfg['data']['neuron_contrast']
    set_bbox(layer)
    layer = viewer.add_labels(a1, name='crystal', opacity=.7, colormap={1: 'red', None: [0, 0, 0, 0]},
                              blending='additive', scale=reader.resolution)
    layer.iso_gradient_mode = 'smooth'
    layer = viewer.add_labels(a2, name='neuron', opacity=.7, colormap={1: 'green', None: [0, 0, 0, 0]},
                              blending='additive', scale=reader.resolution)
    layer.iso_gradient_mode = 'smooth'
    set_camera()

    viewer.screenshot(path=str(repo / "pairs_all.png"), canvas_only=True)
    viewer.close()

    n = np.unique(pairs['neuron'])
    centers = neurons_df.loc[n, ['z', 'y', 'x']].to_numpy()

    # numbers
    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_image(img_n, opacity=.7, blending='additive', scale=reader.resolution)
    layer.contrast_limits = cfg['data']['neuron_contrast']
    layer = viewer.add_labels(lab_n, opacity=.7, blending='additive', scale=reader.resolution, rendering='translucent')
    layer.iso_gradient_mode = 'smooth'
    viewer.add_points(
        neurons_df[['z', 'y', 'x']],
        features={'no': np.arange(len(neurons_df.index)) + 1},
        text={
            'string': '{no}',
            'size': 6,
            'scaling': True,
            'color': 'white',
            'translation': np.array([0, -20, 0]),
        },
        size=10,
        face_color='red', scale=reader.resolution,
    )
    viewer.camera.zoom = 5
    viewer.camera.angles = (0, 0, 90)
    show_scale_bar()

    viewer.screenshot(path=str(repo / "numbers.png"), canvas_only=True)
    viewer.scale_bar.visible = False
    viewer.screenshot(path=str(repo / "numbers_no_scale.png"), canvas_only=True)
    viewer.close()