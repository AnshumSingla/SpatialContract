SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world",
                    "array_orientation": "LIA"}


def process(in_path, out_path):
    import nibabel as nib
    from nibabel.processing import conform

    img = nib.load(in_path)
    nib.save(conform(img, out_shape=(128, 128, 128), voxel_size=(2.0, 2.0, 2.0), order=0, orientation="LIA"), out_path)
