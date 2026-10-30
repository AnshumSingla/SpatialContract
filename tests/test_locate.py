"""Oracle-guided localisation on a tiny fake library: one helper mishandles the affine only for some input orientations."""
import textwrap

from spatialcontract import locate
from spatialcontract.contract import load_from_source

LIB = {
    "fakelib/__init__.py": "",
    "fakelib/io.py": '''
        import nibabel as nib
        import numpy as np

        def load(path):
            img = nib.load(path)
            return np.asarray(img.dataobj), img.affine

        def save(data, affine, path):
            nib.save(nib.Nifti1Image(data, affine), path)
    ''',
    "fakelib/geometry.py": '''
        import nibabel as nib
        import numpy as np

        def canonical(data, affine):
            ornt = nib.orientations.io_orientation(affine)
            if ornt[0, 1] < 0:
                return _flip_first_axis(data, affine)
            return data, affine

        def _flip_first_axis(data, affine):
            return data[::-1].copy(), affine

        def describe(data):
            return {"shape": data.shape, "max": float(data.max())}
    ''',
}

DRIVER = '''
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}
from fakelib import geometry, io

def process(in_path, out_path):
    data, affine = io.load(in_path)
    geometry.describe(data)
    data, affine = geometry.canonical(data, affine)
    io.save(data, affine, out_path)
'''


def test_trace_ranks_the_buggy_helper_first(tmp_path):
    lib = tmp_path / "lib"
    for rel, src in LIB.items():
        (lib / rel).parent.mkdir(parents=True, exist_ok=True)
        (lib / rel).write_text(textwrap.dedent(src))
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    contract = load_from_source(DRIVER)

    loc = locate.locate(driver, contract, lib, seeds=(0, 1, 2), pythonpath=[lib])
    assert loc.n_fail and loc.n_pass, loc.statuses           # some phantoms flip the first axis, some do not
    top = loc.suspects[0]
    assert (top.file, top.qualname) == ("fakelib/geometry.py", "_flip_first_axis")
    assert top.pass_runs == 0 and top.returned_affine
    assert "geometry.py" in loc.summary(3)

    h = locate.hits(loc, [("fakelib/geometry.py", "_flip_first_axis")])
    assert h["func@1"] and h["rank_first_true_function"] == 1
