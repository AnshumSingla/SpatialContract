import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Ensure output array orientation is RAS
    current_ori = o.io_orientation(img.affine)
    target_ori = o.axcodes2ornt(o.axcodes_from_string("RAS"))
    transform = o.ornt_transform(current_ori, target_ori)
    if not np.allclose(transform, np.eye(4)):
        img = img.apply_transform(transform)
    nib.save(img, out_path)
