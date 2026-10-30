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
    aff = img.affine
    vox = 2.0
    # compute world coordinates of the 8 corners of the input voxel grid
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world_corners = (aff @ corners.T).T[:, :3]
    lo = world_corners.min(axis=0)
    hi = world_corners.max(axis=0)
    # output shape to cover the same physical extent with isotropic voxels
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    # output affine: axes aligned with world axes, origin at lo
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    # generate output voxel indices (i,j,k) with 'ij' indexing
    ii, jj, kk = np.meshgrid(np.arange(shape[0]),
                             np.arange(shape[1]),
                             np.arange(shape[2]),
                             indexing='ij')
    out_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                        np.ones_like(ii.ravel())], axis=0)
    # world coordinates of output voxel centers
    world_pts = out_aff @ out_idx
    # map world points to continuous input voxel coordinates
    in_idx_cont = np.linalg.inv(aff) @ world_pts
    # nearest neighbor lookup
    in_idx = np.rint(in_idx_cont[:3]).astype(int)
    # keep only those inside the input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    # reshape with C order (default) to match meshgrid indexing
    out_img = nib.Nifti1Image(out.reshape(shape), out_aff)
    nib.save(out_img, out_path)
