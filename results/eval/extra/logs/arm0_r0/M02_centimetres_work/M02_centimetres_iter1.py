import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Ensure data array is RAS oriented
    orig_ornt = nib.io_orientation(img.affine)
    target_ornt = o.axcodes2ornt('RAS')
    transform = o.ornt_transform(orig_ornt, target_ornt)
    img = img.apply_orientation(transform)
    aff = img.affine.copy()
    # Convert from input units (assumed mm) to meters as per contract
    aff[:3, :] *= 1e-2
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)