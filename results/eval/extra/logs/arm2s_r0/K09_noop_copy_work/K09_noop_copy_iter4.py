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
    data = img.get_fdata()  # guaranteed copy
    aff = img.affine.copy()
    cur_codes = o.aff2axcodes(aff)
    desired_codes = ('R', 'A', 'S')
    if cur_codes != desired_codes:
        transform = o.ornt_transform(o.axcodes2ornt(cur_codes),
                                     o.axcodes2ornt(desired_codes))
        data = o.apply_orientation(data, transform)
        # adjust affine for new voxel ordering
        aff = aff @ np.linalg.inv(o.inv_ornt_aff(transform, data.shape))
    nib.save(nib.Nifti1Image(data, aff), out_path)