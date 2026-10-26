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
    # Current orientation as integer codes
    cur_ornt = o.io_orientation(img.affine)
    # Desired orientation: RAS
    ras_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    # Reorient if needed
    if not np.array_equal(cur_ornt, ras_ornt):
        img = img.as_reoriented(ras_ornt)
    nib.save(img, out_path)
