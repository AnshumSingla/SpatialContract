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
    # Determine current orientation and transform to RAS
    ornt = o.io_orientation(aff)
    target_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    xfm = o.ornt_transform(ornt, target_ornt)
    new_data = o.apply_orientation(data, xfm)
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    nib.save(nib.Nifti1Image(new_data, new_aff), out_path)
