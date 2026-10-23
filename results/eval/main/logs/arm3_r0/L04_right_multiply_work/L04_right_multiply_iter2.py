import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = img.get_fdata()
    affine = img.affine
    # Flip X and Y axes to convert from RAS+ to LPS+
    data = np.flip(data, axis=0)
    data = np.flip(data, axis=1)
    # Build transformation matrix for RAS->LPS (flip X,Y)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    new_affine = ras_to_lps @ affine
    nib.save(nib.Nifti1Image(data, new_affine), out_path)
