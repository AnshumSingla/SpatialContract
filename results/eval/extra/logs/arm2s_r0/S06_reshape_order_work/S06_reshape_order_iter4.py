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
    # indices along each axis from 0 to shape (inclusive) to get voxel lattice corners
    xs = np.arange(data.shape[0] + 1)
    ys = np.arange(data.shape[1] + 1)
    zs = np.arange(data.shape[2] + 1)
    # grid of corners
    ix, iy, iz = np.meshgrid(xs, ys, zs, indexing='ij')
    corners_idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel(), np.ones_like(ix.ravel())], axis=1)
    world_corners = aff @ corners_idx.T
    lo = world_corners[:3].min(axis=1)
    hi = world_corners[:3].max(axis=1)
    shape_out = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    ii, jj, kk = np.indices(shape_out)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape_out)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape_out, order="F"), out_aff), out_path)