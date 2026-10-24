"""Does the UPSTREAM fix (POST) pass the held-out validation on each real case, and does the buggy version (PRE) fail it?
If POST fails validation the held-out checks would be unfair to a repair agent. No LLM, no tokens.
  python scripts/check_real_gold.py [case_id ...]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import heldout, realbugs  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402


def main() -> int:
    want = set(sys.argv[1:])
    bad = 0
    for c in realbugs.CASES:
        if want and c.id not in want:
            continue
        contract = load_from_source(c.script.read_text(encoding="utf-8"))
        out = {}
        for which in ("pre", "post"):
            h = heldout.validate(c.script, contract, run_seed=0, python=str(realbugs.ENVS[c.env]), pythonpath=c.pythonpath(which), hidden_seeds=c.hidden_seeds or None)
            out[which] = h
        print(f"{c.id:24s} PRE passed={out['pre']['passed']} {out['pre']['summary']}", flush=True)
        print(f"{'':24s} POST passed={out['post']['passed']} {out['post']['summary']}", flush=True)
        if not out["post"]["passed"] or out["pre"]["passed"]:
            bad += 1
            for k, v in out["post"]["failing_detail"].items():
                print("   POST failing part:", k, v.replace("\n", " | ")[:400])
    print("cases where the gold fix does not validate cleanly (or PRE validates):", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
