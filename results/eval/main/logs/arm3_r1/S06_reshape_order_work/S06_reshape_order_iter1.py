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
    # world coordinates of input voxel centres
    ijk = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                    for j in (0, data.shape[1] - 1)
                    for k in (0, data.shape[2] - 1)])
    world = (aff @ ijk.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    # number of voxels needed to cover the range at vox resolution
    shape = np.ceil((hi - lo) / vox).astype(int)
    # affine for output: vox mm isotropic, origin at lo (first voxel centre)
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    # generate output voxel index grid
    ii, jj, kk = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    out_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                        np.ones_like(ii.ravel())], axis=0)  # 4 x N
    # world coordinates of output voxel centres
    world_pts = out_aff @ out_idx
    # map back to input voxel indices (continuous)
    in_idx_cont = np.linalg.inv(aff) @ world_pts
    in_idx = np.rint(in_idx_cont[:3]).astype(int)
    # keep only those that lie inside the input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape, order="F"), out_aff), out_path)