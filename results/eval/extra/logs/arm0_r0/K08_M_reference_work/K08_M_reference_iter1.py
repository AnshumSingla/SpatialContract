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
    # Reorient to RAS+
    ras_aff = img.affine
    current_ornt = o.io_orientation(ras_aff)
    ras_ornt = o.axcodes2ornt(o.aff2axcodes(ras_aff))
    # Transform to RAS
    transform_ornt = o.ornt_transform(current_ornt, ras_ornt)
    ras_img = img.apply_transform(transform_ornt)
    # Now affine is in RAS (still in original units, likely mm)
    aff = ras_img.affine.copy()
    # Convert units to meters (contract units = m)
    aff[:3, :] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(ras_img.dataobj), aff), out_path)