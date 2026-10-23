"""For each single phantom seed: how many seeded bugs does the oracle catch and how many controls false-alarm?
No LLM, no API. Writes results/seeded_per_single_phantom.json.  Usage: per_phantom_robustness.py [n_seeds]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import seeded  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out, min_err = {}, {}
    for s in range(n):
        rows = seeded.evaluate(seeds=(s,), workers=6)
        sm = seeded.summarize(rows)
        out[s] = {k: sm[k] for k in ("bugs_detected", "false_alarms", "abstain_correct", "missed_bugs", "false_alarm_ids")}
        for r in rows:
            if r["kind"] == "bug" and r["correct"] and r["worst_error_mm"] is not None and not r["id"].startswith("R02"):
                min_err[r["id"]] = min(min_err.get(r["id"], 1e18), r["worst_error_mm"])
        print(s, out[s], flush=True)
    smallest = sorted(min_err.items(), key=lambda kv: kv[1])[:5]
    print("smallest worst-landmark errors among detected bugs (mm):", smallest)
    (ROOT / "results" / "seeded_per_single_phantom.json").write_text(
        json.dumps({"per_seed": out, "smallest_detected_error_mm": smallest}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
