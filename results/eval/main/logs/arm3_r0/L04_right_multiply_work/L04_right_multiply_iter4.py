import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert from RAS+ world coordinates to LPS+ by reflecting X and Y axes
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    new_affine = ras_to_lps @ img.affine
    nib.save(nib.Nifti1Image(img.get_fdata(), new_affine), out_path)
