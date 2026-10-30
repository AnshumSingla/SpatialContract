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
    # Reorient to RAS+ (closest canonical)
    ras_img = nib.as_closest_canonical(img)
    data = np.asarray(ras_img.dataobj)
    aff = ras_img.affine

    shape = np.array(data.shape)
    # Voxel center indices: 0 .. shape-1
    ijk = np.transpose(np.ogrid[:shape[0], :shape[1], :shape[2]])
    xyz_h = np.concatenate([ijk, np.ones_like(ijk[..., :1])], axis=-1).reshape(-1, 4)
    world_pts = (aff @ xyz_h.T).T[:, :3]  # (N,3)
    lo = world_pts.min(axis=0)
    hi = world_pts.max(axis=0)

    vox = 2.0
    out_shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    ii, jj, kk = np.indices(out_shape)
    out_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones_like(ii).ravel()], axis=0)
    world_out = out_aff @ out_idx  # (4, N)
    # Map world to input voxel indices (continuous)
    inv_aff = np.linalg.inv(aff)
    cont_idx = (inv_aff @ world_out)[:3].T  # (N,3)
    in_idx = np.rint(cont_idx).astype(int)
    valid = np.all((in_idx >= 0) & (in_idx < shape), axis=1)
    out = np.zeros(np.prod(out_shape), dtype=data.dtype)
    out[valid] = data[in_idx[valid, 0], in_idx[valid, 1], in_idx[valid, 2]]
    out = out.reshape(out_shape)

    nib.save(nib.Nifti1Image(out, out_aff), out_path)
