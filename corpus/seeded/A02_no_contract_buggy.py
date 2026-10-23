# seeded case A02_no_contract_buggy: stale-affine bug, no contract declared
import numpy as np
import nibabel as nib
from nibabel import orientations as o

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))
    new = o.apply_orientation(data, xfm)
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    nib.save(nib.Nifti1Image(new, aff), out_path)
