"""Equal gating, recomputed from existing rows with no tokens. An objection: the oracle arm skips the model when the oracle
already passes, while the unit-test arms called the model even when every test passed. Here every arm gets the same rule: if its
own check passes on the original script, the model is not called and the script is left unchanged (no feedback arm: no check,
so no gate). Writes results/eval/gated.json.

  python scripts/analyze_gating.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev, oracle, repo_tests  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402


def main() -> int:
    manifest = {m["id"]: m for m in ev.load_manifest()}
    rows = [json.loads(l) for f in ("main", "extra") for l in (ROOT / f"results/eval/{f}/raw.jsonl").read_text(encoding="utf-8").splitlines()]
    passes = {}                       # (arm, id) -> does the arm's own check pass on the ORIGINAL script?
    for cid, m in manifest.items():
        path = ev.CORPUS / m["file"]
        for arm, strong in (("arm2", False), ("arm2s", True)):
            passes[(arm, cid)] = repo_tests.run_repo_tests(path, m["base"], m.get("spec") or {}, strong=strong).passed
        contract = load_from_source(path.read_text(encoding="utf-8"))
        passes[("arm3", cid)] = contract is not None and not contract.problems() and oracle.check(path, contract).passed
    out = {}
    for arm in ("arm0", "arm2", "arm2s", "arm3"):
        rs = [r for r in rows if r["arm"] == arm]
        gated = []
        for r in rs:
            g = dict(r)
            if passes.get((arm, r["id"]), False):         # the check passes: no model call, script unchanged
                g.update(changed=False, abstained=False, repaired=manifest[r["id"]]["kind"] != "bug" or False)
                if manifest[r["id"]]["kind"] == "bug":
                    g["repaired"] = False
            gated.append(g)
        bugs = [g for g in gated if g["kind"] == "bug"]
        ctrl = [g for g in gated if g["kind"] == "control"]
        abst = [g for g in gated if g["kind"] == "abstain"]
        per_case = ev.per_case(gated, arm, "repaired", {g["id"] for g in bugs})
        out[arm] = {
            "repaired_pct": round(100 * sum(per_case.values()) / len(per_case)),
            "controls_modified": sum(bool(g["changed"]) for g in ctrl), "controls_broken":
                sum(bool(g["changed"]) and not (g["visible_pass"] and g["heldout_pass"]) for g in ctrl), "controls_runs": len(ctrl),
            "abstain_cases_patched": sum(bool(g["changed"]) for g in abst), "abstain_runs": len(abst),
            "check_passes_on_bugs": sum(passes.get((arm, i), False) for i, m in manifest.items() if m["kind"] == "bug"),
            "check_passes_on_controls": sum(passes.get((arm, i), False) for i, m in manifest.items() if m["kind"] == "control"),
        }
        print(f"{arm:6s} repaired {out[arm]['repaired_pct']}% | controls modified {out[arm]['controls_modified']}/{len(ctrl)} "
              f"(broke {out[arm]['controls_broken']}) | must-abstain cases patched {out[arm]['abstain_cases_patched']}/{len(abst)} | "
              f"check passes on {out[arm]['check_passes_on_bugs']} bugs, {out[arm]['check_passes_on_controls']} controls")
    (ROOT / "results/eval/gated.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
