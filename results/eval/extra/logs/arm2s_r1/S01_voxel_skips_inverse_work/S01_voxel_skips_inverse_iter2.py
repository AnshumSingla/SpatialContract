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
    # 8 corners of the input volume (voxel indices)
    corners_idx = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                            for j in (0, data.shape[1] - 1)
                            for k in (0, data.shape[2] - 1)])
    # voxel centers are at index + 0.5 in each dimension
    center_idx = corners_idx.copy()
    center_idx[:, :3] += 0.5
    world_centers = (aff @ center_idx.T).T[:, :3]
    lo_c = world_centers.min(axis=0)
    hi_c = world_centers.max(axis=0)
    shape = np.ceil((hi_c - lo_c) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo_c
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx
    # map world points back to input voxel indices (center convention)
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts).astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)