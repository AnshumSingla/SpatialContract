"""Apply the no-op check (heldout.is_noop) retroactively to every accepted repair. No tokens.

A repair fails it when the original code changed the array's extent and the patch returns its input unchanged on every visible phantom. Seeded repairs are
checked from their saved final patches; library repairs by applying their saved diff to a fresh copy of the pre-fix library.
Library runs whose verdict changes are rewritten in their raw.jsonl with the earlier verdict kept in ``outcome_before_noop_check``.

  python scripts/check_noop.py      -> results/noop_check.json"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev, heldout, libagent, libmutate, realbugs  # noqa: E402


def main() -> int:
    manifest = {m["id"]: m for m in ev.load_manifest()}
    seeded = []
    for r in map(json.loads, (ROOT / "results/eval/main/raw.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["arm"] == "arm3" and r["kind"] == "bug" and r["repaired"]:
            p = ROOT / f"results/eval/main/logs/arm3_r{r['rep']}/{r['id']}_work/{r['id']}_iter{r['iterations']}.py"
            flagged = heldout.is_noop(p) and heldout.changes_extent(ev.CORPUS / manifest[r["id"]]["file"])
            seeded.append({"id": r["id"], "rep": r["rep"], "noop": flagged})
    cases = {c.id: c for c in realbugs.CASES}
    lib = []
    for run_id in ("real3", "real4", "staged1", "staged2", "mut1"):
        f = ROOT / f"results/real_repair/{run_id}/raw.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        changed = False
        for row in rows:
            if row["outcome"] != "REPAIRED":
                continue
            case = cases[row["case"]]
            name = row["case"] if run_id.startswith("mut") else f"{row['case']}_r{row['rep']}"
            ws = libagent.Workspace(case)
            try:
                if run_id.startswith("mut"):          # whole-file mutants: regenerate the winning mutant (deterministic) instead of patching
                    target, label = row["winner"].split(" ", 1)
                    rel, q = target.split("::")
                    src = ws._resolve(rel)
                    src.write_text(dict(libmutate.mutants(src.read_text(encoding="utf-8"), q))[label], encoding="utf-8")
                else:
                    subprocess.run(["git", "apply", "-p1", "--directory=lib", "--unsafe-paths",
                                    str(ROOT / f"results/real_repair/{run_id}/logs/{name}.diff")], cwd=ws.root, check=True, capture_output=True)
                py = str(realbugs.ENVS[case.env])
                flagged = heldout.is_noop(case.script, py, ws.pythonpath) and heldout.changes_extent(case.script, py, case.pythonpath("pre"))
            finally:
                ws.cleanup()
            lib.append({"run": f"{run_id}/{name}", "noop": flagged})
            if flagged:
                row["outcome_before_noop_check"], row["outcome"] = row["outcome"], "OVERFIT"
                row.setdefault("reasons", []).append("no-op: the patch makes the output equal the input (the pipeline's work was removed); "
                                                     "caught by the no-op check added after this run")
                changed = True
        if changed:
            f.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    out = {"seeded_repairs": len(seeded), "seeded_flagged": [s for s in seeded if s["noop"]],
           "library_repairs": len(lib), "library_flagged": [x for x in lib if x["noop"]], "seeded": seeded, "library": lib}
    (ROOT / "results/noop_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k not in ("seeded", "library")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
