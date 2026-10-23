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
    resampled = nproc.resample_to_output(img, 2.0, order=0)
    # Preserve the original affine to keep spatial alignment
    resampled_img = nib.Nifti1Image(resampled.get_fdata(), img.affine, img.header)
    nib.save(resampled_img, out_path)