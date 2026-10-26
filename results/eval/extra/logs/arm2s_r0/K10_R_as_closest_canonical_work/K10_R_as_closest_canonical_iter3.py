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
    # Current orientation: voxels -> world mapping per voxel axis
    cur_ornt = o.io_orientation(img.affine)
    # Desired orientation: RAS
    ras_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    # Compute transformation if needed
    if not np.array_equal(cur_ornt, ras_ornt):
        transform = o.ornt_transform(cur_ornt, ras_ornt)
        img = img.apply_orientation(transform)
    nib.save(img, out_path)
