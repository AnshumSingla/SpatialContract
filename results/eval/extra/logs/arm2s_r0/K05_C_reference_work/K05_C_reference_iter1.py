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
    if nz.size == 0:
        # No positive voxels: save empty image with original affine
        nib.save(nib.Nifti1Image(data, aff), out_path)
        return
    lo = np.maximum(nz.min(axis=0) - 3, 0)
    hi = np.minimum(nz.max(axis=0) + 4, data.shape)
    cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    shift = np.eye(4, dtype=aff.dtype)
    shift[:3, 3] = lo
    new_aff = aff @ shift
    nib.save(nib.Nifti1Image(cropped, new_aff), out_path)