import numpy as np
import nibabel as nib
from nibabel import orientations as o
import nibabel.processing as nproc

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to RAS (closest canonical) to satisfy world_frame RAS
    img = nib.as_closest_canonical(img)
    resampled = nproc.resample_to_output(img, 2.0, order=0)
    nib.save(resampled, out_path)
