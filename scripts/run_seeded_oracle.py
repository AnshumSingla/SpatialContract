"""Build corpus/seeded and run the deterministic oracle over it (no LLM, no API). Writes results/seeded_oracle.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import seeded  # noqa: E402


def main() -> int:
    seeded.build()
    rows = seeded.evaluate()
    summ = seeded.summarize(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "seeded_oracle.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    for r in rows:
        mark = "ok " if r["correct"] else "BAD"
        print(f"{mark} {r['id']:28s} expect={r['expect']:7s} got={r['verdict']:7s} worst_mm={r['worst_error_mm']} "
              f"lost={r['landmark_lost']} ori_viol={r['orientation_violation']}")
    print(json.dumps(summ, indent=1))
    return 0 if not summ["missed_bugs"] and not summ["false_alarm_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
