"""Does a repair keep the pipeline's output extent? Measured on every accepted repair, no tokens.

A mutation search "repaired" TorchIO's Crop by deleting the crop: every landmark stays in place, so the oracle passed it. To see
whether "a repair must not change the output's array shape relative to the original code" is a safe extra check, this compares, on
the visible phantoms, the output shape of the original (buggy) script and of each accepted patch: exactly, and as a sorted tuple
(an axis-order fix legitimately permutes the shape).

  python scripts/check_extent.py      -> results/extent_check.json"""
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev, oracle  # noqa: E402
from spatialcontract.phantom import make_phantom  # noqa: E402


def shapes(code: Path, seeds=(0, 1, 2)):
    out = []
    with tempfile.TemporaryDirectory(prefix="sc_extent_") as d:
        d = Path(d)
        for s in seeds:
            ph = make_phantom(s)
            i, o = d / f"in{s}.nii.gz", d / f"out{s}.nii.gz"
            ph.save(i)
            ok, _ = oracle._run_code(code.resolve(), i, o, d)
            out.append(tuple(nib.load(str(o)).shape) if ok and o.exists() else None)
    return out


def main() -> int:
    manifest = {m["id"]: m for m in ev.load_manifest()}
    rows = []
    for r in map(json.loads, (ROOT / "results/eval/main/raw.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["arm"] != "arm3" or r["kind"] != "bug" or not r["repaired"]:
            continue
        orig = shapes(ev.CORPUS / manifest[r["id"]]["file"])
        patched = shapes(ROOT / f"results/eval/main/logs/arm3_r{r['rep']}/{r['id']}_work/{r['id']}_iter{r['iterations']}.py")
        same = all(a == b for a, b in zip(orig, patched) if a is not None)
        same_sorted = all(sorted(a) == sorted(b) for a, b in zip(orig, patched) if a is not None and b is not None)
        rows.append({"id": r["id"], "rep": r["rep"], "original": orig, "patched": patched, "same_shape": same, "same_sorted_shape": same_sorted,
                     "original_errored": any(a is None for a in orig)})
    out = {"repairs": len(rows), "same_shape": sum(r["same_shape"] for r in rows), "same_sorted_shape": sum(r["same_sorted_shape"] for r in rows),
           "differ": [(r["id"], r["original"][0], r["patched"][0]) for r in rows if not r["same_sorted_shape"]], "rows": rows}
    (ROOT / "results/extent_check.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
