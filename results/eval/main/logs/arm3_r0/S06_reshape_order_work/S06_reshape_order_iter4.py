import numpy as np
import nibabel as nib
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

    # World bounding box of input volume (voxel centres)
    shape_in = np.array(data.shape)
    corners = np.array([[i, j, k, 1.0]
                        for i in (0, shape_in[0] - 1)
                        for j in (0, shape_in[1] - 1)
                        for k in (0, shape_in[2] - 1)])
    world_corners = (aff @ corners.T).T[:, :3]
    lo = world_corners.min(axis=0)
    hi = world_corners.max(axis=0)

    vox = 2.0
    out_shape = np.ceil((hi - lo) / vox).astype(int) + 1
    # Output voxel centre coordinates
    xs = lo[0] + np.arange(out_shape[0]) * vox
    ys = lo[1] + np.arange(out_shape[1]) * vox
    zs = lo[2] + np.arange(out_shape[2]) * vox
    ii, jj, kk = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()])  # 3 × N

    # Output affine: index -> world = lo + [i,j,k]*vox
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    # Map output voxel centres to input voxel coordinates (continuous)
    world_pts = coords  # already world centres
    inv_aff = np.linalg.inv(aff)
    inp_coords = inv_aff[:3, :3] @ world_pts + inv_aff[:3, 3:4]

    # Sample with linear interpolation
    out = map_coordinates(data, inp_coords, order=1, mode='constant', cval=0.0)
    out = out.reshape(ii.shape)

    nib.save(nib.Nifti1Image(out, out_aff), out_path)
