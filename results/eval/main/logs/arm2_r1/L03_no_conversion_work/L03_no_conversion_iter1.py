import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert from RAS+ (default nibabel) to LPS+
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    new_affine = ras_to_lps @ img.affine
    out_img = nib.Nifti1Image(np.asarray(img.dataobj), new_affine)
    nib.save(out_img, out_path)