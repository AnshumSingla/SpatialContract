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
    # Ensure data array is mutable
    data = np.asarray(img.dataobj)
    aff = img.affine

    # Reorient to RAS+ (world_frame RAS)
    orig_ornt = o.io_orientation(img.header)
    ras_ornt = o.axcodes2ornt('RAS')
    transform = o.ornt_transform(orig_ornt, ras_ornt)
    data = o.apply_orientation(data, transform)
    aff = o.apply_orientation(aff, transform)  # affine transforms similarly

    # Determine bounding box in world coordinates (voxel centers)
    shape = np.array(data.shape)
    # indices of voxel centers: 0..shape-1
    ijk = np.transpose(np.ogrid[:shape[0], :shape[1], :shape[2]])
    xyz = (aff @ np.concatenate([ijk, np.ones_like(ijk[..., :1])], axis=-1).reshape(-1, 4).T).T[:, :3]
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)

    # Desired isotropic voxel size
    vox = 2.0
    # Output shape covering the same physical extent
    out_shape = np.ceil((hi - lo) / vox).astype(int) + 1
    # Output affine: vox isotropic, origin at lo
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    # Create output grid indices
    ii, jj, kk = np.indices(out_shape)
    out_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones_like(ii).ravel()], axis=0)
    # World coordinates of output voxel centers
    world_pts = out_aff @ out_idx  # shape (4, N)
    # Map world to input voxel indices (continuous)
    inv_aff = np.linalg.inv(aff)
    cont_idx = (inv_aff @ world_pts)[:3].T  # shape (N, 3)
    # Nearest-neighbor sampling
    in_idx = np.rint(cont_idx).astype(int)
    # Valid mask
    valid = np.all((in_idx >= 0) & (in_idx < shape), axis=1)
    out = np.zeros(np.prod(out_shape), dtype=data.dtype)
    out[valid] = data[in_idx[valid, 0], in_idx[valid, 1], in_idx[valid, 2]]
    out = out.reshape(out_shape)

    nib.save(nib.Nifti1Image(out, out_aff), out_path)
