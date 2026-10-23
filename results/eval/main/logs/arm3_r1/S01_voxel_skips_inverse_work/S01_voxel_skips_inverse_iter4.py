import numpy as np
import nibabel as nib
from nibabel import orientations as o
from scipy.ndimage import map_coordinates

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
    out_aff[:3, 3] = lo - 0.5 * vox  # translation so that first voxel center is at lo
    # output voxel indices
    oi = np.arange(shape[0])
    oj = np.arange(shape[1])
    ok = np.arange(shape[2])
    oo, pp, qq = np.meshgrid(oi, oj, ok, indexing='ij')
    # voxel center coordinates in output space
    out_idx_c = np.stack([oo.ravel() + 0.5, pp.ravel() + 0.5, qq.ravel() + 0.5, np.ones_like(oo).ravel()], axis=0)  # 4 x M
    out_world = (out_aff @ out_idx_c).T[:, :3]  # M x 3
    # map world to input voxel coordinates (continuous)
    inv_aff = np.linalg.inv(aff)
    homo = np.hstack([out_world, np.ones((out_world.shape[0], 1))])  # M x 4
    in_homo = (inv_aff @ homo.T).T  # M x 4
    # voxel index continuous = in_homo[:,:3] - 0.5 (since aff maps index+0.5 to world)
    coord = in_homo[:, :3] - 0.5  # shape (M,3)
    # map_coordinates expects coordinates in (z, y, x) order? Actually expects (dim0, dim1, dim2) matching array shape.
    # Our data shape is (nx, ny, nz) corresponding to i,j,k.
    # coord columns are (i_cont, j_cont, k_cont)
    # map_coordinates takes coordinates as list of arrays per dimension, in order of array dimensions.
    coords = [coord[:, 2], coord[:, 1], coord[:, 0]]  # (k, j, i) if data shape is (i,j,k)?? Wait: data.shape[0] is i (x?), but we don't know orientation.
    # Safer: transpose to match data shape order.
    # Let's reshape coord to match data shape order by reversing.
    # Assuming nibabel follows (x,y,z) ordering in data array? Actually nibabel dataobj shape corresponds to (i,j,k) where i is first dimension (usually left-right).
    # We'll keep as is and let map_coordinates handle.
    # Provide coordinates in same order as data shape: dim0, dim1, dim2.
    coords_list = [coord[:, 0], coord[:, 1], coord[:, 2]]
    samples = map_coordinates(data, coords_list, order=0, mode='constant', cval=0)
    out = samples.reshape(shape)
    nib.save(nib.Nifti1Image(out, out_aff), out_path)
