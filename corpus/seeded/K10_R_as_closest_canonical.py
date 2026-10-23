# seeded case K10_R_as_closest_canonical: library call nib.as_closest_canonical
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
    nib.save(nib.as_closest_canonical(img), out_path)
