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
    # voxel center indices for input bounding box
    i_c = np.arange(data.shape[0]) + 0.5
    j_c = np.arange(data.shape[1]) + 0.5
    k_c = np.arange(data.shape[2]) + 0.5
    ii, jj, kk = np.meshgrid(i_c, j_c, k_c, indexing='ij')
    centers = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones_like(ii).ravel()], axis=0)  # 4 x N
    world_centers = (aff @ centers).T[:, :3]  # N x 3
    lo = world_centers.min(axis=0)
    hi = world_centers.max(axis=0)
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    # output voxel indices (no offset)
    oi = np.arange(shape[0])
    oj = np.arange(shape[1])
    ok = np.arange(shape[2])
    oo, pp, qq = np.meshgrid(oi, oj, ok, indexing='ij')
    out_idx = np.stack([oo.ravel(), pp.ravel(), qq.ravel(), np.ones_like(oo).ravel()], axis=0)  # 4 x M
    out_world = (out_aff @ out_idx).T[:, :3]  # M x 3
    # map world to input voxel coordinates (continuous)
    inv_aff = np.linalg.inv(aff)
    homo = np.hstack([out_world, np.ones((out_world.shape[0], 1))])  # M x 4
    in_homo = (inv_aff @ homo.T).T  # M x 4
    # voxel index continuous = in_homo[:,:3] - 0.5 (since aff maps index+0.5 to world)
    in_idx_cont = in_homo[:, :3] - 0.5
    in_idx = np.rint(in_idx_cont).astype(int)
    ok_mask = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)), axis=1)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok_mask] = data[in_idx[ok_mask, 0], in_idx[ok_mask, 1], in_idx[ok_mask, 2]]
    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)
