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
    # Ensure the output inherits the original header (including sform/qform)
    resampled_header = img.header.copy()
    resampled_header.set_data_shape(resampled.shape)
    resampled_header.set_zooms(list(resampled.header.get_zooms()[:3]) + [resampled.header.get_zooms()[3]] if resampled.header.get_zooms().size > 3 else list(resampled.header.get_zooms()))
    # Actually simpler: copy affine and use original header
    out_img = nib.Nifti1Image(resampled.get_fdata(), img.affine, img.header)
    nib.save(out_img, out_path)