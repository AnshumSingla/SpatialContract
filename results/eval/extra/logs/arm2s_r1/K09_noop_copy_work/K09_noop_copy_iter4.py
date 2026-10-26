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
    # Determine if already RAS
    axcodes = nib.aff2axcodes(aff)
    if axcodes == ('R', 'A', 'S'):
        data = np.asarray(img.dataobj)
        new_aff = aff
    else:
        # Orientation of current voxels to RAS
        orig_ornt = o.io_orientation(aff)
        target_ornt = o.axcodes2ornt(('R', 'A', 'S'))
        trans = o.ornt_transform(orig_ornt, target_ornt)
        # Apply orientation to data (copy)
        data = o.apply_orientation(np.asarray(img.dataobj), trans)
        # Compute new affine for oriented data
        move = o.inv_ornt_aff(trans, img.shape)
        new_aff = np.dot(aff, move)
    nib.save(nib.Nifti1Image(data, new_aff), out_path)