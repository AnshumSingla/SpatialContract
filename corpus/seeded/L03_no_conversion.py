# seeded case L03_no_conversion: forgets the RAS->LPS conversion and saves the input affine
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
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), img.affine), out_path)
