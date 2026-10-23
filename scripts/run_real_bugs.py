"""Run the deterministic oracle on every real-bug case at its pre-fix and post-fix commit (no LLM, no API).
Usage: run_real_bugs.py [case_id ...]   Writes results/real_bugs_oracle.json (only when run without arguments)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import realbugs  # noqa: E402


def main() -> int:
    want = set(sys.argv[1:])
    rows = []
    for c in realbugs.CASES:
        if want and c.id not in want:
            continue
        r = realbugs.check_case(c)
        rows.append(r)
        print(f"{c.id:24s} pre={r['pre']['verdict']:8s} post={r['post']['verdict']:8s} "
              f"pre_worst_mm={r['pre'].get('worst_error_mm')} reproduced={r['reproduced']}", flush=True)
        for which in ("pre", "post"):
            if r[which]["verdict"] in ("ERROR", "NOT_RUN"):
                print("   ", which, r[which].get("feedback") or r[which].get("reason"))
    if not want:
        realbugs.save_results(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
