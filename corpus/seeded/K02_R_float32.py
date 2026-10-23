# seeded case K02_R_float32: reference plus float32 cast and contiguous copy
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
    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))
    new = o.apply_orientation(data, xfm)
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    new = np.ascontiguousarray(new, dtype=np.float32)
    nib.save(nib.Nifti1Image(new, new_aff), out_path)
