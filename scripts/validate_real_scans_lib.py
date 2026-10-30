"""Every repair INSIDE a real library (agent, staged and mutation runs), on real scans, next to the pre-fix and the upstream-fixed library. No LLM, no tokens.

For every REPAIRED run in results/real_repair/real3 and real4, the agent's diff is applied to a fresh copy of the pre-fix library and
the case's driver is run on every real scan; the same driver also runs on the pre-fix and upstream-fixed worktrees. Each output
is compared with the input at the same physical points (realscan.consistency), and the agent's output is also compared directly
with the upstream fix's output (same grid and affine, and voxelwise equality).

  python scripts/validate_real_scans_lib.py      -> results/real_scans_lib.json"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import libagent, oracle, realbugs, realscan  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402


def run_output(case, img, python, pythonpath):
    """The driver's raw output on a real scan: (data, affine) or None."""
    with tempfile.TemporaryDirectory(prefix="sc_reallib_") as d:
        d = Path(d)
        src, out = d / "scan.nii.gz", d / "out.nii.gz"
        nib.save(img, str(src))
        ok, _ = oracle._run_code(case.script, src, out, d, python, pythonpath)
        if not ok:
            return None
        o = nib.load(str(out))
        return np.asarray(o.dataobj).astype(np.float32), np.asarray(o.affine, dtype=float)


def main() -> int:
    scans = realscan.scans()
    cases = {c.id: c for c in realbugs.CASES}
    runs = []
    only = set(sys.argv[1:])                       # optional run ids: validate only those and merge into the existing results
    for run_id in ("real3", "real4", "staged1", "staged2", "mut1"):
        if only and run_id not in only:
            continue
        f = ROOT / f"results/real_repair/{run_id}/raw.jsonl"
        if not f.exists():
            continue
        for row in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if row["outcome"] == "REPAIRED":
                name = row["case"] if run_id.startswith("mut") else f"{row['case']}_r{row['rep']}"
                runs.append((run_id, row["case"], row["rep"], ROOT / f"results/real_repair/{run_id}/logs/{name}.diff"))
    out_rows = []
    for run_id, cid, rep, diff in runs:
        case = cases[cid]
        contract = load_from_source(case.script.read_text(encoding="utf-8"))
        py = str(realbugs.ENVS[case.env])
        ws = libagent.Workspace(case)
        try:
            ap = subprocess.run(["git", "apply", "-p1", "--directory=lib", "--unsafe-paths", str(diff)], cwd=ws.root, capture_output=True, text=True)
            if ap.returncode:
                out_rows.append({"run": f"{run_id}/{cid}_r{rep}", "error": "diff did not apply: " + ap.stderr[:200]})
                continue
            for name, img in scans:
                r = {"run": f"{run_id}/{cid}_r{rep}", "case": cid, "scan": name}
                for label, pp in (("pre", case.pythonpath("pre")), ("upstream", case.pythonpath("post")), ("agent", ws.pythonpath)):
                    c = realscan.consistency(case.script, contract, img, python=py, pythonpath=pp)
                    r[label] = {k: c.get(k) for k in ("r", "covered_fraction", "orientation_ok", "out_axcodes", "verdict", "detail") if k in c}
                a, u = run_output(case, img, py, ws.pythonpath), run_output(case, img, py, case.pythonpath("post"))
                if a is None or u is None:
                    r["agent_equals_upstream"] = None
                else:
                    same_grid = a[0].shape == u[0].shape and np.allclose(a[1], u[1], atol=1e-4)
                    r["agent_equals_upstream"] = bool(same_grid and np.allclose(a[0], u[0], atol=1e-3 * (np.abs(u[0]).max() + 1)))
                    r["agent_vs_upstream_max_affine_diff_mm"] = None if a[0].shape != u[0].shape else round(float(np.abs(a[1] - u[1]).max()), 5)
                out_rows.append(r)
                print(f"{r['run']:34s} {name:24s} pre r={r['pre'].get('r')} orient={r['pre'].get('orientation_ok')} | "
                      f"upstream r={r['upstream'].get('r')} orient={r['upstream'].get('orientation_ok')} | agent r={r['agent'].get('r')} "
                      f"orient={r['agent'].get('orientation_ok')} | agent==upstream {r['agent_equals_upstream']}", flush=True)
        finally:
            ws.cleanup()
    dest = ROOT / "results/real_scans_lib.json"
    if only and dest.exists():
        out_rows = [r for r in json.loads(dest.read_text(encoding="utf-8")) if r["run"].split("/")[0] not in only] + out_rows
    dest.write_text(json.dumps(out_rows, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
