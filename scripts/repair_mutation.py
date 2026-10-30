"""Model-free repair of the real library bugs (libmutate.py: mutation search over the trace's top suspects). No tokens.

  python scripts/repair_mutation.py --run-id mut1 [--cases ...] [--k 6] [--max-candidates 600]
Same result fields as scripts/repair_real.py and scripts/repair_staged.py (results/real_repair/<run-id>/raw.jsonl)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import libmutate, realbugs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-candidates", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--run-id", default=time.strftime("mut-%Y%m%d-%H%M%S"))
    a = ap.parse_args()
    out = ROOT / "results" / "real_repair" / a.run_id
    (out / "logs").mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = {r["case"] for r in map(json.loads, raw.read_text(encoding="utf-8").splitlines())} if raw.exists() else set()
    for case in realbugs.CASES:
        if (a.cases and case.id not in a.cases) or case.id in done:
            continue
        res = libmutate.repair_mutation(case, k=a.k, max_candidates=a.max_candidates, workers=a.workers,
                                        log_path=out / "logs" / f"{case.id}.jsonl")
        row = res.to_dict() | {"rep": 0}
        (out / "logs" / f"{case.id}.diff").write_text(res.diff, encoding="utf-8")
        with raw.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(f"{case.id:24s} {res.outcome:9s} candidates={res.candidates} screened_pass={res.screened_pass} winner={res.winner} "
              f"localized={res.localized} {res.seconds}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
