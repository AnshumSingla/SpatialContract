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
    new_affine = img.affine @ ras_to_lps
    # Copy header and set the sform/qform to the new affine
    new_header = img.header.copy()
    new_header.set_sform(new_affine, code=nib.Nifti1Header.SCANNER_ANAT)
    new_header.set_qform(new_affine, code=nib.Nifti1Header.SCANNER_ANAT)
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), new_affine, header=new_header), out_path)