import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    nz = np.argwhere(data > 0)
    lo = np.maximum(nz.min(axis=0) - 3, 0)
    hi = np.minimum(nz.max(axis=0) + 4, data.shape)
    cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    shift = np.eye(4)
    shift[:3, 3] = lo
    new_aff = aff @ shift
    # Flip Z axis of data and adjust affine accordingly
    flip_z = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, cropped.shape[2] - 1],
                       [0, 0, 0, 1]])
    final_aff = new_aff @ flip_z
    nib.save(nib.Nifti1Image(cropped[:, :, ::-1], final_aff), out_path)