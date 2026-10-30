SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world",
                    "array_orientation": "RAS"}


def process(in_path, out_path):
    import nibabel as nib
    import numpy as np
    import torch
    from monai.data import MetaTensor
    from monai.data.utils import orientation_ras_lps
    from monai.transforms import Orientation

    img = nib.load(in_path)
    data = torch.as_tensor(np.asarray(img.dataobj), dtype=torch.float32)[None]
    affine_lps = orientation_ras_lps(torch.as_tensor(img.affine, dtype=torch.float64))
    mt = MetaTensor(data, affine=affine_lps, meta={"space": "LPS"})
    out = Orientation(axcodes="RAS")(mt)
    affine_ras = orientation_ras_lps(out.affine).numpy()
    nib.save(nib.Nifti1Image(out[0].numpy(), affine_ras), out_path)
