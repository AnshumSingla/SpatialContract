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
    # Determine current orientation codes
    cur_codes = o.aff2axcodes(aff)
    # Desired orientation is RAS
    desired_codes = ('R', 'A', 'S')
    if cur_codes != desired_codes:
        # Transform to RAS
        transform = o.axcodes2ornt(cur_codes, desired_codes)
        data = o.apply_orientation(data, transform)
        aff = o.apply_orientation(aff, transform)
    nib.save(nib.Nifti1Image(data, aff), out_path)