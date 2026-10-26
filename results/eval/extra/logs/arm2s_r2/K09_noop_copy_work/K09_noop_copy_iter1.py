import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to RAS (right-anterior-superior)
    orig_ornt = o.io_orientation(img.affine)
    ras_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    transform = o.ornt_transform(orig_ornt, ras_ornt)
    data = np.asarray(img.dataobj)
    data = o.apply_orientation(data, transform)
    aff = o.apply_orientation(img.affine, transform)
    nib.save(nib.Nifti1Image(data, aff), out_path)