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
    # Keep the resampled voxel size (2mm) but align the origin with the input
    aligned_affine = resampled.affine.copy()
    aligned_affine[:3, 3] = img.affine[:3, 3]
    out_img = nib.Nifti1Image(resampled.get_fdata(), aligned_affine, img.header)
    nib.save(out_img, out_path)