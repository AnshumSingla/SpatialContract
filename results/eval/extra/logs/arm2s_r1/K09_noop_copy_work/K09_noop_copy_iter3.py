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
    aff = img.affine
    axcodes = nib.aff2axcodes(aff)
    if axcodes == ('R', 'A', 'S'):
        # Already RAS, just copy
        data = np.asarray(img.dataobj)
        new_aff = aff
    else:
        # Compute orientation transform to RAS
        orig_ornt = o.io_orientation(aff)
        target_ornt = o.axcodes2ornt(('R', 'A', 'S'))
        trans = o.ornt_transform(orig_ornt, target_ornt)
        # Apply orientation to data
        data = o.apply_orientation(img.get_fdata(), trans)
        # Update affine accordingly
        inv = o.inv_ornt_trans(trans)
        new_aff = np.dot(inv, aff)
    nib.save(nib.Nifti1Image(data, new_aff), out_path)