import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    import nibabel.processing as nproc
    img = nib.load(in_path)
    # Ensure input is in RAS orientation
    img = nib.as_closest_canonical(img)
    resampled = nproc.resample_to_output(img, 2.0, order=0)
    nib.save(resampled, out_path)