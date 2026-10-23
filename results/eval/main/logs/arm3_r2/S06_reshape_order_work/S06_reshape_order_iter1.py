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
    # compute bounding box in world space (voxel centers)
    shape_in = np.array(data.shape)
    corners = np.array([[i, j, k, 1.0] for i in (0, shape_in[0]-1)
                        for j in (0, shape_in[1]-1) for k in (0, shape_in[2]-1)])
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    # output grid spacing
    out_shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    # voxel indices of output grid
    ii, jj, kk = np.indices(out_shape)
    out_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones_like(ii).ravel()], axis=0)
    # world coordinates of output voxel centers
    world_pts = out_aff @ out_idx
    # map back to input voxel indices (continuous)
    in_idx_cont = np.linalg.inv(aff) @ world_pts
    in_idx = np.rint(in_idx_cont[:3]).astype(int)
    # valid indices
    ok = np.all((in_idx >= 0) & (in_idx < shape_in[:, None]), axis=0)
    out = np.zeros(np.prod(out_shape), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(out_shape), out_aff), out_path)
