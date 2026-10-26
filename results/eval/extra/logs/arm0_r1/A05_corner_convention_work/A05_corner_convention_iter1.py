import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_corner_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    # Convert affine from voxel-center to voxel-corner
    # corner = center - 0.5 * voxel_size_in_world
    voxel_size_vec = np.diag(aff[:3, :3])  # assumes no shearing; for general case use first three columns
    offset = aff[:3, :3] @ (0.5 * voxel_size_vec)
    aff_corner = aff.copy()
    aff_corner[:3, 3] -= offset
    nz = np.argwhere(data > 0)
    lo = np.maximum(nz.min(axis=0) - 3, 0)
    hi = np.minimum(nz.max(axis=0) + 4, data.shape)
    cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    shift = np.eye(4)
    shift[:3, 3] = lo
    new_aff = aff_corner @ shift
    nib.save(nib.Nifti1Image(cropped, new_aff), out_path)