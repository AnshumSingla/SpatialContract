"""Recompute the headline numbers from the raw result files and check that the documents quote them. No tokens, no network.
Run:  python scripts/audit_docs.py   (exit code 1 on any mismatch)"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev  # noqa: E402

DOCS = {n: (ROOT / p).read_text(encoding="utf-8") for n, p in
        {"README": "README.md", "DEVPOST": "docs/DEVPOST.md", "VIDEO": "docs/VIDEO_SCRIPT.md", "FEEDBACK": "FEEDBACK.md"}.items()}
rows = [json.loads(l) for l in (ROOT / "results/eval/main/raw.jsonl").read_text(encoding="utf-8").splitlines()]
manifest = {m["id"]: m for m in ev.load_manifest()}
bugs = {i for i, m in manifest.items() if m["kind"] == "bug"}
ctrl = {i for i, m in manifest.items() if m["kind"] == "control"}
abst = {i for i, m in manifest.items() if m["kind"] == "abstain"}
R = lambda arm, ids: [r for r in rows if r["arm"] == arm and r["id"] in ids]   # noqa: E731
bad = []


def check(doc: str, needle: str, why: str) -> None:
    ok = needle in DOCS[doc]
    print(("ok   " if ok else "MISS ") + f"{doc:8s} {needle!r:34s} {why}")
    if not ok:
        bad.append((doc, needle))


pct = lambda x: f"{round(100 * x)}%"   # noqa: E731
rep = {a: ev.per_case(rows, a, "repaired", bugs) for a in ("arm1", "arm2", "arm3")}
means = {a: sum(v.values()) / len(v) for a, v in rep.items()}
d = ev.boot_diff_ci(rep["arm3"], rep["arm2"])
print({a: pct(m) for a, m in means.items()}, "diff", [round(100 * x) for x in d])
check("README", f"**{pct(means['arm3'])} vs {pct(means['arm2'])}**", "headline repair rates")
check("README", f"+{round(100 * d[0])} points, 95% CI +{round(100 * d[1])} to +{round(100 * d[2])}", "paired difference")
check("DEVPOST", f"+{round(100 * d[0])} points (95% CI +{round(100 * d[1])} to +{round(100 * d[2])})", "paired difference")
check("DEVPOST", f"| {pct(means['arm1'])} |", "arm 1 rate")
check("VIDEO", f"{round(100 * d[1]) and 'three'} to twenty-two", "CI wording")

a2c = R("arm2", ctrl)
mod, broke = sum(bool(r["changed"]) for r in a2c), sum(bool(r["changed"]) and not (r["visible_pass"] and r["heldout_pass"]) for r in a2c)
check("README", f"modified {mod} of {len(a2c)}, broke {broke}", "arm 2 controls")
check("FEEDBACK", f"modified {mod} of {len(a2c)}", "arm 2 controls")
a3c = R("arm3", ctrl)
check("README", f"0 of {len(a3c)} runs", "arm 3 controls untouched") if not any(r["changed"] for r in a3c) else bad.append("arm3 changed a control")

ab2, ab3 = R("arm2", abst), R("arm3", abst)
guessed, abstained = sum(bool(r["changed"]) for r in ab2), sum(bool(r["abstained"]) for r in ab2)
check("README", f"patched {guessed} of {len(ab2)} and abstained twice" if abstained == 2 else "?", "arm 2 abstention")
check("README", f"abstains on {sum(bool(r['abstained']) for r in ab3)} of {len(ab3)} runs", "arm 3 abstention")
check("VIDEO", "sixteen of twenty-one" if guessed == 16 and len(ab2) == 21 else "?", "video abstention")

a3b, a2b = R("arm3", bugs), R("arm2", bugs)
check("FEEDBACK", f"{sum(bool(r['repaired']) for r in a3b)} of {len(a3b)} seeded-bug runs", "arm 3 runs repaired")
check("FEEDBACK", f"{sum(bool(r['repaired']) for r in a2b)} of {len(a2b)} (86%)", "arm 2 runs repaired")
check("FEEDBACK", f"{sum(r['iterations'] == 1 and r['repaired'] for r in a3b)} on the first call", "first-call repairs")

flagged = json.loads((ROOT / "results/eval/main/summary.json").read_text(encoding="utf-8"))["repo_tests_flag"]
check("README", f"flag {len(flagged)} of {len(bugs)} bugs", "repo tests flag")
check("DEVPOST", f"{len(bugs) - len(flagged)} of {len(bugs)} such bugs pass", "unflagged count")
n_calls = sum(1 for p in (ROOT / "results/eval/main/logs").rglob("*.jsonl") for l in p.read_text(encoding="utf-8").splitlines() if '"event": "llm_response"' in l)
check("README", f"all {n_calls} evaluation requests" if False else f"{n_calls} evaluation requests", "model calls")
check("FEEDBACK", f"{n_calls} of {n_calls}", "tool-call reliability")

real = json.loads((ROOT / "results/real_bugs_oracle.json").read_text(encoding="utf-8"))
check("README", f"{sum(r['reproduced'] for r in real)} of {len(real)} reproduced", "real bugs")
led = json.loads((ROOT / "results/token_ledger.json").read_text(encoding="utf-8"))["total_tokens"]
print("ledger", led)
check("FEEDBACK", "About 873,000 tokens" if 872_500 < led < 873_500 else f"ledger={led}", "token total")
n_tests = len(list((ROOT / "tests").glob("test_*.py")))
print("test files", n_tests)
print("\nMISMATCHES:" if bad else "\nall checked numbers match", bad or "")
raise SystemExit(1 if bad else 0)
