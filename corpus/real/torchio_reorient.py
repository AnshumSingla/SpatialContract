SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world",
                    "array_orientation": "RAS"}


def process(in_path, out_path):
    import torchio as tio

    subject = tio.Subject(t1=tio.ScalarImage(in_path))
    tio.Reorient(orientation="RAS")(subject).t1.save(out_path)
