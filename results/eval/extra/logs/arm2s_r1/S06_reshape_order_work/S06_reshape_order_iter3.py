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
    # world coordinates of input voxel centers (corners of the voxel grid)
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    # unit basis directions of input affine (assuming orthogonal)
    basis = aff[:3, :3]
    norms = np.linalg.norm(basis, axis=0, keepdims=True)
    unit_basis = basis / norms
    # extents along each basis direction
    lo_i = np.dot(lo, unit_basis.T)  # shape (3,)
    hi_i = np.dot(hi, unit_basis.T)
    shape = np.ceil((hi_i - lo_i) / vox).astype(int) + 1
    # origin so that first voxel center lies at lo
    origin = np.sum(lo_i[:, None] * unit_basis, axis=0)
    out_aff = np.eye(4)
    out_aff[:3, :3] = vox * unit_basis
    out_aff[:3, 3] = origin
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape, order="F"), out_aff), out_path)