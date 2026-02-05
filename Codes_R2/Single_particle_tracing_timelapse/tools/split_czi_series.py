import sys
import numpy as np
from aicsimageio import AICSImage
from pylibCZIrw import czi as pyczi
from pathlib import Path


# Generalized dynamic iteration
def iterate_planes(czi, doc):
    """
    Dynamically iterate through YX planes in an array based on axes_order.

    Args:
        array (numpy.ndarray): The data array.
        axes_order (str): String representing the axes order (e.g., "TCZYX").
    """
    array = czi.data
    axes_order = czi.dims.order
    axes_indices = {axis: i for i, axis in enumerate(axes_order)}
    # Get the positions of Y and X axes
    y_index = axes_indices['Y']
    x_index = axes_indices['X']

    # Identify non-YX axes for iteration
    non_yx_axes = [axis for axis in czi.dims.order if axis not in 'YX']
    non_yx_dims = {axis: array.shape[axes_indices[axis]] for axis in non_yx_axes}

    # Iterate dynamically over non-YX dimensions
    for coord in np.ndindex(*(non_yx_dims[axis] for axis in non_yx_axes)):
        # Build a slicing object for the array
        slices = [slice(None)] * array.ndim
        for i, axis in enumerate(non_yx_axes):
            slices[axes_indices[axis]] = coord[i]
        slices[y_index] = slice(None)  # Keep all Y
        slices[x_index] = slice(None)  # Keep all X
        # Extract the YX plane
        doc.write(
            data=array[tuple(slices)],
            plane={axis: coord[i] for i, axis in enumerate(non_yx_axes)},
            compression_options="zstd0:ExplicitLevel=10"
        )


if __name__ == '__main__':

    # Example usage
    input_czi_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # Load the .czi file
    czi = AICSImage(input_czi_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in czi.scenes:
        # Select the series
        czi.set_scene(s)
        image_data = czi.data

        # Generate filenames
        new_path = output_dir / f'{input_czi_path.stem}_{s}.czi'
        with pyczi.create_czi(str(new_path), exist_ok=True) as czidoc_w:
            iterate_planes(czi, czidoc_w)