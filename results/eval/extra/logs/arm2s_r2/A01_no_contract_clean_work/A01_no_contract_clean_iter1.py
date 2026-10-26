import numpy as np
import nibabel as nib
from nibabel import orientations as o

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    # Transform to RAS orientation
    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))
    new = o.apply_orientation(data, xfm)
    # Compute new affine: apply inverse of orientation transform to original affine
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    # Copy header information (sform/qform) from input
    out_img = nib.Nifti1Image(new, new_aff, header=img.header)
    nib.save(out_img, out_path)