import pytest
from sources import BUGS, CORRECT

from spatialcontract import oracle
from spatialcontract.contract import from_dict

RAS_MM = from_dict(dict(world_frame="RAS", units="mm", affine_convention="voxel_center_to_world",
                        array_orientation="RAS"))


@pytest.fixture
def write(tmp_path):
    def _w(src, name="code.py"):
        p = tmp_path / name
        p.write_text(src)
        return p
    return _w


def test_correct_pipeline_passes(write):
    rep = oracle.check(write(CORRECT), None)
    assert rep.verdict == "PASS", rep.feedback()
    assert rep.worst()[0] < 0.01     # exact for lossless reorientation


@pytest.mark.parametrize("name", sorted(BUGS))
def test_each_bug_class_is_detected(write, name):
    rep = oracle.check(write(BUGS[name]), None)
    assert rep.verdict == "FAIL", f"{name}: {rep.feedback()}"
    err, _, _ = rep.worst()
    assert err > 5.0, f"{name}: worst error only {err} mm"
    assert "landmark" in rep.feedback()


def test_every_bug_fails_on_every_single_phantom(write):
    for name, src in BUGS.items():
        for s in (0, 1, 2, 3):
            assert oracle.check(write(src), None, seeds=(s,)).verdict == "FAIL", (name, s)


def test_noop_pipeline_fails(write):
    src = CORRECT.replace("    nib.save(nib.Nifti1Image(new, new_aff), out_path)\n", "    nib.save(img, out_path)\n")
    assert oracle.check(write(src), None).verdict == "FAIL"


def test_lps_contract_changes_the_verdict(write):
    # Same code, same RAS-mm output: PASS under an RAS contract, FAIL under an LPS contract.
    assert oracle.check(write(CORRECT), RAS_MM).verdict == "PASS"
    lps = from_dict(dict(world_frame="LPS", units="mm", affine_convention="voxel_center_to_world"))
    rep = oracle.check(write(CORRECT), lps)
    assert rep.verdict == "FAIL" and rep.worst()[0] > 5


def test_correct_lps_pipeline_passes_lps_contract(write):
    src = (CORRECT.replace('"world_frame": "RAS"', '"world_frame": "LPS"')
           .replace('    "array_orientation": "RAS",\n', "")
           .replace("nib.Nifti1Image(new, new_aff)", "nib.Nifti1Image(new, np.diag([-1, -1, 1, 1]) @ new_aff)"))
    rep = oracle.check(write(src), None)
    assert rep.verdict == "PASS", rep.feedback()


def test_correct_metre_pipeline_passes_metre_contract(write):
    src = (CORRECT.replace('"units": "mm"', '"units": "m"')
           .replace("nib.Nifti1Image(new, new_aff)", "nib.Nifti1Image(new, np.diag([1e-3, 1e-3, 1e-3, 1]) @ new_aff)"))
    assert oracle.check(write(src), None).verdict == "PASS"


def test_no_contract_abstains(write):
    src = CORRECT.split("SPATIAL_CONTRACT")[0] + "def process(a, b):\n    pass\n"
    rep = oracle.check(write(src), None)
    assert rep.verdict == "ABSTAIN" and rep.phantoms == []     # code never executed


def test_ambiguous_contract_abstains(write):
    src = CORRECT.replace('"voxel_center_to_world"', '"voxel_corner_to_world"')
    assert oracle.check(write(src), None).verdict == "ABSTAIN"
    rep = oracle.check(write(CORRECT.replace('"units": "mm"', '"units": "unknown"')), None)
    assert rep.verdict == "ABSTAIN" and "units" in rep.feedback()


def test_crash_is_error_not_fail(write):
    rep = oracle.check(write(CORRECT.replace("o.io_orientation(aff)", "o.nonexistent(aff)")), None)
    assert rep.verdict == "ERROR" and "ERROR" in rep.feedback()


def test_lost_landmarks_reported(write):
    src = CORRECT.replace("    new = o.apply_orientation(data, xfm)\n",
                          "    new = o.apply_orientation(data, xfm)[:20]\n")
    rep = oracle.check(write(src), None)
    assert rep.verdict == "FAIL" and "LOST" in rep.feedback()


def test_api_key_not_visible_to_code_under_test(write, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "sentinel-secret")
    src = CORRECT.replace("def process(in_path, out_path):\n",
                          "def process(in_path, out_path):\n    import os\n"
                          "    assert 'NEBIUS_API_KEY' not in os.environ\n")
    assert oracle.check(write(src), None).verdict == "PASS"


def test_degenerate_output_affine_is_an_error_verdict_not_a_crash(write):
    for bad in ("np.zeros((4, 4))", "np.full((4, 4), np.nan)", "np.diag([1.0, 0.0, 1.0, 1.0])"):
        src = CORRECT.replace("nib.Nifti1Image(new, new_aff)", f"nib.Nifti1Image(new, {bad})")
        rep = oracle.check(write(src), None)
        assert rep.verdict in ("ERROR", "FAIL"), (bad, rep.feedback())
