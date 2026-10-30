import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine  # maps voxel index (i,j,k) -> world (voxel center)
    
    # Decompose input affine into orientation and scale
    M = aff[:3, :3]
    # scales are column norms (since columns are direction vectors scaled by voxel size)
    scales = np.sqrt(np.sum(M * M, axis=0))
    # avoid division by zero
    scales = np.where(scales == 0, 1, scales)
    orient = M / scales  # unit direction vectors
    
    # Output affine: same orientation, isotropic voxel size 2mm
    vox_size = 2.0
    out_aff = np.eye(4)
    out_aff[:3, :3] = orient * vox_size
    out_aff[:3, 3] = aff[:3, 3]  # keep same world origin (voxel (0,0,0) center)
    
    # Determine output voxel grid that encloses the input's world bounding box
    # World coordinates of input voxel corners (8 corners)
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world_corners = (aff @ corners.T).T[:, :3]
    lo = world_corners.min(axis=0)
    hi = world_corners.max(axis=0)
    
    # Convert world bounds to output voxel coordinates (may be fractional)
    def world_to_vox(world_pt):
        homo = np.append(world_pt, 1.0)
        return np.linalg.inv(out_aff) @ homo
    vox_lo = world_to_vox(lo)
    vox_hi = world_to_vox(hi)
    # Integer voxel indices covering the range
    min_idx = np.floor(vox_lo[:3]).astype(int)
    max_idx = np.ceil(vox_hi[:3]).astype(int)
    shape = max_idx - min_idx + 1
    
    # Adjust output affine so that voxel index (0,0,0) corresponds to min_idx
    translation = out_aff @ np.append(min_idx, 1.0)
    out_aff[:3, 3] = translation[:3]
    
    # Create output voxel indices
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T  # 4 x N
    
    # Map output voxel -> input voxel (homogeneous)
    M_map = np.linalg.inv(aff) @ out_aff
    in_idx_homog = M_map @ out_idx
    in_idx = np.rint(in_idx_homog[:3]).astype(int)
    
    # Keep only those that map inside the input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    
    # Linear indexing into input data (Fortran order as nibabel uses)
    lin_idx = (in_idx[0, ok] +
               in_idx[1, ok] * data.shape[0] +
               in_idx[2, ok] * data.shape[0] * data.shape[1])
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data.flat[lin_idx]
    
    # Save output image (Fortran order reshape)
    nib.save(nib.Nifti1Image(out.reshape(shape, order='F'), out_aff), out_path)
