"""Brain-extraction preprocessing step.

Loads a scanner NIfTI, reorients it to RAS+ (closest canonical orientation) so that downstream code can index
voxels as (right, anterior, superior), and saves it. World coordinates stay in millimetres.
"""
import nibabel as nib


def process(in_path, out_path):
    img = nib.load(in_path)
    canonical = nib.as_closest_canonical(img)
    nib.save(canonical, out_path)
