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
    # Determine current orientation codes
    cur_ornt = o.io_orientation(img.affine)
    # Desired orientation is RAS
    ras_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    # Transform if needed
    if not np.array_equal(cur_ornt, ras_ornt):
        transform = o.ornt_transform(cur_ornt, ras_ornt)
        img = img.apply_transform(transform)
    # Ensure affine uses voxel_center_to_world (default nibabel)
    nib.save(img, out_path)
