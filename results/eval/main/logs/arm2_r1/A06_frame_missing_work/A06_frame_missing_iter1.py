import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), img.affine), out_path)