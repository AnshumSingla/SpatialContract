"""Adoption path: decorator / check_function on in-memory functions, the pytest fixture, and the CLI. No network, no LLM."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from sources import BUGS, CORRECT

from spatialcontract import api, cli, oracle
from spatialcontract.api import NotImportable, check_function, spatial_contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import inmemory_example as ex  # noqa: E402

pytest_plugins = ["spatialcontract.pytest_plugin"]
SUMMARY_FAIL = re.compile(r"^FAIL  worst landmark \d+\.\d+ mm off \(phantom seed \d+, label \d+\)$")


def reorient_ras(img):
    return nib.as_closest_canonical(img)


@spatial_contract(world_frame="RAS", units="mm")
def boom(img):
    raise ValueError("boom")


@spatial_contract(world_frame="RAS", units="mm")
def env_probe(img):
    assert "NEBIUS_API_KEY" not in os.environ
    return img


@pytest.mark.parametrize("fn, verdict", [(ex.crop_buggy, "FAIL"), (ex.crop_fixed, "PASS"),
                                         (ex.crop_array_buggy, "FAIL"), (ex.crop_array_fixed, "PASS")])
def test_example_crops_isolated(fn, verdict):
    rep = check_function(fn)
    assert rep.verdict == verdict, rep.feedback()
    if verdict == "FAIL":
        assert rep.worst()[0] > 5 and SUMMARY_FAIL.match(api.summary(rep)), api.summary(rep)
    else:
        assert rep.worst()[0] < 0.01


def test_decorator_returns_the_function_unchanged():
    assert ex.crop_fixed.__name__ == "crop_fixed" and not hasattr(ex.crop_fixed, "__wrapped__")
    d = ex.crop_array_fixed.__spatial_contract__
    assert d.signature == "array" and d.contract.world_frame == "RAS" and d.contract.affine_convention == "voxel_center_to_world"


def test_in_process_matches_isolated_measurement():
    a = check_function(ex.crop_buggy, seeds=(1,))
    b = check_function(ex.crop_buggy, seeds=(1,), isolate=False)
    assert a.verdict == b.verdict == "FAIL" and a.worst() == pytest.approx(b.worst())
    assert oracle._run_code.__name__ == "_run_code"       # runner restored


def test_undecorated_function_with_keyword_contract():
    assert check_function(reorient_ras, world_frame="RAS", units="mm", array_orientation="RAS").verdict == "PASS"
    assert check_function(reorient_ras, contract={"world_frame": "LPS", "units": "mm"}, seeds=(0,)).verdict == "FAIL"


def test_missing_contract_abstains_without_running():
    calls = []
    rep = check_function(lambda img: calls.append(1) or img, units="mm", isolate=False)
    assert rep.verdict == "ABSTAIN" and "world_frame" in rep.feedback() and calls == [] and rep.phantoms == []
    assert check_function(reorient_ras).verdict == "ABSTAIN"


def test_not_importable_is_a_clear_error_and_isolate_false_runs_it():
    def local(img):
        return nib.as_closest_canonical(img)
    for fn in (local, lambda img: img):
        with pytest.raises(NotImportable, match="isolate=False"):
            check_function(fn, world_frame="RAS", units="mm")
    assert check_function(local, world_frame="RAS", units="mm", isolate=False, seeds=(0,)).verdict == "PASS"


def test_exception_is_error_in_both_modes():
    for isolate in (True, False):
        rep = check_function(boom, seeds=(0,), isolate=isolate)
        assert rep.verdict == "ERROR" and "boom" in rep.feedback() and api.summary(rep).startswith("ERROR  phantom seed 0")


def test_wrong_return_type_is_error():
    rep = check_function(lambda d, a: d, world_frame="RAS", units="mm", signature="array", isolate=False, seeds=(0,))
    assert rep.verdict == "ERROR" and "(array, affine)" in rep.feedback()


def test_array_signature_accepts_bool_and_int64_outputs():
    kw = dict(world_frame="RAS", units="mm", signature="array", isolate=False, seeds=(0,))
    rep = check_function(lambda d, a: (d.astype(np.int64), a), **kw)     # a no-op copy: positions are right
    assert rep.verdict == "PASS" and rep.worst()[0] < 0.01
    rep = check_function(lambda d, a: (d == 1, a), **kw)                   # saved, analysed: the other labels are lost
    assert rep.verdict == "FAIL" and "LOST" in rep.feedback()


def test_subprocess_does_not_see_secrets(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "sentinel-secret")
    assert check_function(env_probe, seeds=(0,)).verdict == "PASS"
    assert check_function(env_probe, seeds=(0,), isolate=False).verdict == "ERROR"   # in-process: no isolation


def test_assert_spatial_contract_and_fixture(spatial_oracle):
    with pytest.raises(AssertionError, match=r"FAIL  worst landmark .* mm off[\s\S]*landmark \d: expected"):
        api.assert_spatial_contract(ex.crop_buggy, seeds=(0,))
    assert spatial_oracle(ex.crop_fixed, seeds=(0,)).passed


def test_fixture_failure_message_in_a_real_pytest_run(tmp_path):
    (tmp_path / "test_inner.py").write_text(
        "import sys\nsys.path.insert(0, %r)\nimport inmemory_example as ex\n\n"
        "def test_fixed(spatial_oracle):\n    spatial_oracle(ex.crop_fixed, seeds=(0,))\n\n"
        "def test_buggy(spatial_oracle):\n    spatial_oracle(ex.crop_buggy, seeds=(0,))\n" % str(ROOT / "examples"))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "spatialcontract.pytest_plugin", "-p", "no:cacheprovider",
                        str(tmp_path / "test_inner.py")], capture_output=True, text=True, cwd=tmp_path, timeout=120,
                       env=os.environ | {"PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 1 and "1 failed, 1 passed" in r.stdout, r.stdout + r.stderr
    assert re.search(r"FAIL  worst landmark \d+\.\d mm off \(phantom seed 0, label \d\)", r.stdout)
    assert "landmark 1: expected" in r.stdout and "pytest_plugin.py" not in r.stdout


def test_cli_exit_codes_and_summary(tmp_path, capsys):
    good, bad, bare = tmp_path / "good.py", tmp_path / "bad.py", tmp_path / "bare.py"
    good.write_text(CORRECT)
    bad.write_text(BUGS["stale_affine"])
    bare.write_text("def process(a, b):\n    raise SystemExit('never run')\n")
    assert cli.main(["check", str(good), "--seeds", "0"]) == 0
    assert cli.main(["check", str(bad), "--seeds", "1"]) == 1
    out = capsys.readouterr().out
    assert re.search(r"^FAIL  worst landmark \d+\.\d mm off \(phantom seed 1, label \d\)  \[", out, re.M), out
    assert cli.main(["check", str(bare)]) == cli.EXIT["ABSTAIN"] == 4
    assert "ABSTAIN" in capsys.readouterr().out
    err = tmp_path / "err.py"
    err.write_text(CORRECT.replace("o.io_orientation(aff)", "o.nonexistent(aff)"))
    assert cli.main(["check", str(err), str(good), "--seeds", "0"]) == 3
    assert cli.main(["check", str(err), str(bad), str(bare), "--seeds", "0"]) == 1    # FAIL outranks ERROR and ABSTAIN
    (tmp_path / "syntax.py").write_text("def process(:\n")
    assert cli.main(["check", str(tmp_path / "syntax.py")]) == 3
    assert "ERROR  syntax error" in capsys.readouterr().out


def test_cli_decorated_functions_and_json(tmp_path, capsys):
    path = str(ROOT / "examples" / "inmemory_example.py")
    assert cli.main(["check", path + "::crop_fixed", "--seeds", "0", "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())[0]["verdict"] == "PASS"
    capsys.readouterr()
    assert cli.main(["check", path, "--seeds", "0", "--json"]) == 1
    rows = json.loads(capsys.readouterr().out)
    assert {r["target"].split("::")[1]: r["verdict"] for r in rows} == {
        "crop_buggy": "FAIL", "crop_fixed": "PASS", "crop_array_buggy": "FAIL", "crop_array_fixed": "PASS"}
    assert rows[0]["report"]["phantoms"][0]["landmarks"]


def test_cli_reads_decorator_statically(tmp_path, capsys):
    src = ("from spatialcontract import spatial_contract\nimport spatialcontract as sc\nUNITS = 'mm'\n\n"
           "@sc.spatial_contract(world_frame='LPS', units=UNITS)\ndef f(img):\n    return img\n\n"
           "@spatial_contract\ndef g(img):\n    return img\n")
    (tmp_path / "mod.py").write_text(src)
    assert cli.main(["check", str(tmp_path / "mod.py")]) == 4
    out = capsys.readouterr().out
    assert out.count("ABSTAIN  spatial contract unusable: missing field") == 2     # non-literal units; bare decorator


def test_cli_module_entry_point(tmp_path):
    (tmp_path / "good.py").write_text(CORRECT)
    r = subprocess.run([sys.executable, "-m", "spatialcontract.cli", "check", str(tmp_path / "good.py"), "--seeds", "0"],
                       capture_output=True, text=True, timeout=120, env=os.environ | {"PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0 and r.stdout.startswith("PASS  worst landmark"), r.stdout + r.stderr
