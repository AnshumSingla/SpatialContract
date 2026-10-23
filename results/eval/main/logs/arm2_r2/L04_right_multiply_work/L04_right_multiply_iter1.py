import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert RAS to LPS by flipping X and Y axes
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    data = np.asarray(img.dataobj)
    # Flip data along X and Y to match the affine change
    data = data[::-1, ::-1, :]
    new_affine = img.affine @ ras_to_lps
    nib.save(nib.Nifti1Image(data, new_affine, img.header), out_path)