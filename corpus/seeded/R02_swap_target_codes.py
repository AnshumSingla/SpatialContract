# seeded case R02_swap_target_codes: target codes (A,R,S) instead of (R,A,S): world is right, array order violates contract
import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("A", "R", "S")))
    new = o.apply_orientation(data, xfm)
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    nib.save(nib.Nifti1Image(new, new_aff), out_path)
