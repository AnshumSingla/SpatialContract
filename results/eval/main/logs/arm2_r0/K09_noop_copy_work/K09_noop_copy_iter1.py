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
    data = np.asarray(img.dataobj)
    aff = img.affine
    # Reorient to RAS if needed
    orient = o.io_orientation(aff)
    ras_orient = o.axcodes2oriento(o.aff2axcodes(nib.load(in_path).affine))  # dummy
    # Actually compute target orientation for RAS
    ras_orient = o.axcodes2oriento(('R', 'A', 'S'))
    # Transform
    transform = o.ornt_transform(orient, ras_orient)
    if not np.allclose(transform, np.eye(4)):
        data = o.apply_orientation(data, transform)
        aff = o.apply_orientation(aff, transform)
    nib.save(nib.Nifti1Image(data, aff), out_path)