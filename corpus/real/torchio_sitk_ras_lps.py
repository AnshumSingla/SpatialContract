SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}


def process(in_path, out_path):
    import torchio as tio

    tio.ScalarImage(in_path).save(out_path)
