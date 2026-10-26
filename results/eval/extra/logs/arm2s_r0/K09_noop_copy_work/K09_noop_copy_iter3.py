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
    cur_codes = o.aff2axcodes(aff)
    desired_codes = ('R', 'A', 'S')
    if cur_codes != desired_codes:
        transform = o.ornt_transform(o.axcodes2ornt(cur_codes),
                                     o.axcodes2ornt(desired_codes))
        data = o.apply_orientation(data, transform).copy()
        aff = aff @ np.linalg.inv(o.inv_ornt_aff(transform, data.shape))
    else:
        data = data.copy()
        aff = aff.copy()
    nib.save(nib.Nifti1Image(data, aff), out_path)