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
    vox = 2.0
    
    # Get voxel center coordinates for all voxels
    ijk = np.indices(data.shape).reshape(3, -1).T
    xyz_centers = (aff @ np.hstack([ijk, np.ones((ijk.shape[0], 1))]).T).T[:, :3]
    
    lo = xyz_centers.min(axis=0)
    hi = xyz_centers.max(axis=0)
    
    # Output affine: RAS aligned, isotropic voxels
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo  # origin at minimum center
    
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    
    # Voxel indices of output grid
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    
    # World coordinates of output voxel centers
    world_pts = out_aff @ out_idx
    
    # Map world points back to input voxel indices (continuous)
    in_idx_cont = np.linalg.inv(aff) @ world_pts
    in_idx = np.rint(in_idx_cont[:3]).astype(int)
    
    # Valid indices inside input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    
    nib.save(nib.Nifti1Image(out.reshape(shape, order="F"), out_aff), out_path)
