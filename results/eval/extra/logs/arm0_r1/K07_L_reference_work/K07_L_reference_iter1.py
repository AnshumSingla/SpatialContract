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
    # Convert affine from whatever frame to LPS
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    new_aff = ras_to_lps @ img.affine
    
    # Determine current orientation and target orientation (LPS)
    cur_ornt = nib.io_orientation(img.affine)
    # LPS orientation codes: (-1,0), (-1,1), (0,2) correspond to x=-i, y=-j, z=k
    target_ornt = np.array([[-1, 0], [-1, 1], [0, 2]])
    # Compute transformation to bring data to target orientation
    transform = nib.ornt_transform(cur_ornt, target_ornt)
    # Apply orientation change to data
    data = nib.apply_orientation(img.get_fdata(), transform)
    
    nib.save(nib.Nifti1Image(data, new_aff), out_path)