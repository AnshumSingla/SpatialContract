import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert voxel-to-world affine from mm to metres
    aff = img.affine.copy()
    aff[:3, :3] *= 1e-3
    
    # Reorient image to RAS+ if needed
    current_ornt = o.io_orientation(aff)
    target_ornt = o.axcodes_to_ornt(('R', 'A', 'S'))
    transform = o.ornt_transform(current_ornt, target_ornt)
    if not np.allclose(transform, np.eye(4)):
        data = o.apply_orientation(np.asarray(img.dataobj), transform)
        aff = o.apply_orientation(aff, transform)
    else:
        data = np.asarray(img.dataobj)
    
    nib.save(nib.Nifti1Image(data, aff), out_path)
