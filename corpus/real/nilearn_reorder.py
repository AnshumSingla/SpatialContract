SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}


def process(in_path, out_path):
    import nibabel as nib
    from nilearn.image import reorder_img

    nib.save(reorder_img(nib.load(in_path), resample="nearest"), out_path)
