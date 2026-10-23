# Real bug: MONAI Spacing(scale_extent=True) computed a wrong output offset for oblique affines (fixed in d76adc73, #5253)
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}


def process(in_path, out_path):
    import nibabel as nib
    import numpy as np
    import torch
    from monai.data import MetaTensor
    from monai.transforms import Spacing

    img = nib.load(in_path)
    data = torch.as_tensor(np.asarray(img.dataobj), dtype=torch.float32)[None]
    mt = MetaTensor(data, affine=torch.as_tensor(img.affine, dtype=torch.float64))
    out = Spacing(pixdim=2.0, mode="nearest", padding_mode="zeros", scale_extent=True)(mt)
    nib.save(nib.Nifti1Image(out[0].numpy(), out.affine.numpy()), out_path)
