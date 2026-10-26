import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to RAS (right-anterior-superior) orientation
    ras_img = o.apply_orientation(img, o.axcodes_to_ornt(o.aff2axcodes(img.affine)), o.axcodes_to_ornt(('R','A','S')))
    # Alternatively simpler: use nibabel.as_closest_canonical then ensure RAS
    # but we directly reorient:
    canonical = nib.as_closest_canonical(img)
    current = o.aff2axcodes(canonical.affine)
    target = ('R','A','S')
    if current != target:
        transform = o.ornt_transform(o.axcodes_to_ornt(current), o.axcodes_to_ornt(target))
        canonical = canonical.as_reordered(transform)
    nib.save(canonical, out_path)