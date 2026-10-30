"""Do the fixes that passed the phantom checks also hold on REAL scans? No LLM, no tokens.

Runs on every real scan from realscan.scans(): the 11 correct seeded controls, the 31 buggy originals, and the final patch of
every seeded-bug run the agent REPAIRED in the main evaluation (arm 3). Each run's output is compared with the input at the same
physical points (realscan.consistency). The pass threshold is calibrated on the correct controls only, never on the bugs:
a run passes on a scan when its correlation is at least the lowest correct control's on that scan minus 0.01, it covers at least
half the input (or as much as the lowest correct control of its family, for crops), and it meets the contract's orientation.

  python scripts/validate_real_scans.py      -> results/real_scans.json"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev, realscan  # noqa: E402
from spatialcontract.contract import SpatialContract, load_from_source  # noqa: E402

MARGIN_R = 0.01


def subjects() -> list[dict]:
    manifest = ev.load_manifest()
    out = [{"kind": m["kind"], "id": m["id"], "base": m["base"], "bug_class": m.get("bug_class"), "path": ev.CORPUS / m["file"]}
           for m in manifest if m["kind"] in ("control", "bug")]
    base = {m["id"]: m for m in manifest}
    for r in map(json.loads, (ROOT / "results/eval/main/raw.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["arm"] == "arm3" and r["kind"] == "bug" and r["repaired"]:
            p = ROOT / f"results/eval/main/logs/arm3_r{r['rep']}/{r['id']}_work/{r['id']}_iter{r['iterations']}.py"
            out.append({"kind": "repaired_patch", "id": f"{r['id']}#r{r['rep']}", "base": base[r["id"]]["base"],
                        "bug_class": base[r["id"]].get("bug_class"), "path": p})
    return out


def main() -> int:
    scans = realscan.scans()
    print("scans:", [n for n, _ in scans], flush=True)
    subs = subjects()
    tasks = [(s, n, img) for s in subs for n, img in scans]

    def work(t):
        s, name, img = t
        contract = load_from_source(Path(s["path"]).read_text(encoding="utf-8")) or SpatialContract()
        return {"kind": s["kind"], "id": s["id"], "base": s["base"], "bug_class": s["bug_class"], "scan": name,
                **realscan.consistency(s["path"], contract, img)}

    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(work, tasks))

    # calibrate on the correct controls only
    ctrl = [r for r in rows if r["kind"] == "control"]
    r_min = {n: min(r["r"] for r in ctrl if r["scan"] == n and r.get("r") is not None) for n, _ in scans}
    cov_min = {(b, n): min(r["covered_fraction"] for r in ctrl if r["base"] == b and r["scan"] == n)
               for b in {r["base"] for r in ctrl} for n, _ in scans}
    for r in rows:
        if r.get("verdict") == "ERROR":
            continue
        ok_r = r.get("r") is not None and r["r"] >= r_min[r["scan"]] - MARGIN_R
        ok_cov = r["covered_fraction"] >= min(0.5, cov_min.get((r["base"], r["scan"]), 0.5) - 0.02)
        r["verdict"] = "PASS" if ok_r and ok_cov and r["orientation_ok"] is not False else "FAIL"

    def summary(kind):
        rs = [r for r in rows if r["kind"] == kind]
        ids = sorted({r["id"] for r in rs})
        all_pass = [i for i in ids if all(r["verdict"] == "PASS" for r in rs if r["id"] == i)]
        any_fail = [i for i in ids if any(r["verdict"] in ("FAIL", "ERROR") for r in rs if r["id"] == i)]
        return {"n": len(ids), "pass_on_every_scan": len(all_pass), "fail_on_some_scan": len(any_fail),
                "runs": len(rs), "runs_pass": sum(r["verdict"] == "PASS" for r in rs), "failing_ids": any_fail}

    out = {"scans": [n for n, _ in scans], "threshold_r": {n: round(v - MARGIN_R, 4) for n, v in r_min.items()},
           "summary": {k: summary(k) for k in ("control", "bug", "repaired_patch")}, "rows": rows}
    (ROOT / "results/real_scans.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in out["summary"].items():
        print(f"{k:15s} {v['pass_on_every_scan']}/{v['n']} pass on every scan; {v['fail_on_some_scan']} fail on some scan "
              f"({v['runs_pass']}/{v['runs']} runs pass)  {v['failing_ids'][:12]}")
    print("thresholds:", out["threshold_r"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
