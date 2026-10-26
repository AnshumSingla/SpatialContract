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
check("README", f"oracle feedback **{pct(means['arm3'])}**", "headline repair rate")
check("README", f"Oracle vs original tests: +{round(100 * d[0])} points (95% CI +{round(100 * d[1])} to +{round(100 * d[2])})", "paired difference")
check("DEVPOST", f"+{round(100 * d[0])} points (95% CI +{round(100 * d[1])} to +{round(100 * d[2])})", "paired difference")
check("DEVPOST", f"| {pct(means['arm1'])} |", "arm 1 rate")

a2c = R("arm2", ctrl)
mod, broke = sum(bool(r["changed"]) for r in a2c), sum(bool(r["changed"]) and not (r["visible_pass"] and r["heldout_pass"]) for r in a2c)
check("README", f"original tests {mod} (broke {broke})", "arm 2 controls")
check("FEEDBACK", f"modified {mod} of {len(a2c)}", "arm 2 controls")
a3c = R("arm3", ctrl)
check("README", f"0 of {len(a3c)} runs", "arm 3 controls untouched") if not any(r["changed"] for r in a3c) else bad.append("arm3 changed a control")

ab2, ab3 = R("arm2", abst), R("arm3", abst)
guessed, abstained = sum(bool(r["changed"]) for r in ab2), sum(bool(r["abstained"]) for r in ab2)
check("README", f"original tests {guessed}, stronger" if abstained == 2 else "?", "arm 2 abstention")
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
ledger_file = ROOT / "results/token_ledger.json"      # gitignored: absent in a fresh clone
if ledger_file.exists():
    led = json.loads(ledger_file.read_text(encoding="utf-8"))["total_tokens"]
    print("ledger", led)
    check("FEEDBACK", f"About {led / 1e6:.1f} million tokens", "token total")
else:
    print("SKIP token total: results/token_ledger.json is not in the repository (it is gitignored)")
# ---- extra baselines (results/eval/extra)
ex = [json.loads(l) for l in (ROOT / "results/eval/extra/raw.jsonl").read_text(encoding="utf-8").splitlines()]
E = lambda arm, ids: [r for r in ex if r["arm"] == arm and r["id"] in ids]   # noqa: E731
rep0, rep2s = ev.per_case(ex, "arm0", "repaired", bugs), ev.per_case(ex, "arm2s", "repaired", bugs)
m0, m2s = sum(rep0.values()) / len(rep0), sum(rep2s.values()) / len(rep2s)
d2s = ev.boot_diff_ci(rep["arm3"], rep2s)
print("arm0", pct(m0), "arm2s", pct(m2s), "arm3-arm2s", [round(100 * x) for x in d2s])
check("README", f"no feedback {pct(m0)}", "arm 0 repair rate")
check("README", f"stronger unit-test suite {pct(m2s)}**", "arm 2s repair rate")
check("README", f"**+{round(100 * d2s[0])} point (CI {round(100 * d2s[1])} to +{round(100 * d2s[2])})", "arm 3 - arm 2s")
for arm, label in (("arm0", "no feedback"), ("arm2s", "stronger tests")):
    c, a = E(arm, ctrl), E(arm, abst)
    mod_, broke_ = sum(bool(r["changed"]) for r in c), sum(bool(r["changed"]) and not (r["visible_pass"] and r["heldout_pass"]) for r in c)
    check("README", f"{label} {mod_} of {len(c)} (broke {broke_})" if arm == "arm0" else f"stronger tests {mod_} (broke {broke_})", f"{arm} controls")
    check("DEVPOST", f"{mod_} of {len(c)} (broke {broke_})", f"{arm} controls in Devpost")
    check("README", f"{label} {sum(bool(r['changed']) for r in a)} of {len(a)}" if arm == "arm0" else f"stronger tests {sum(bool(r['changed']) for r in a)}", f"{arm} abstention")
n_extra = sum(1 for p in (ROOT / "results/eval/extra/logs").rglob("*.jsonl") for l in p.read_text(encoding="utf-8").splitlines() if '"event": "llm_response"' in l)
print("extra model calls", n_extra)
check("FEEDBACK", f"{n_extra} model calls", "extra baseline calls")
check("DEVPOST", f"| {pct(m0)} |", "arm 0 in Devpost")
check("DEVPOST", f"| {pct(m2s)} |", "arm 2s in Devpost")

n_tests = len(list((ROOT / "tests").glob("test_*.py")))
print("test files", n_tests)
# ---- library repair (results/real_repair/real3)
lib = [json.loads(l) for l in (ROOT / "results/real_repair/real3/raw.jsonl").read_text(encoding="utf-8").splitlines()]
nrep = sum(r["outcome"] == "REPAIRED" for r in lib)
nloc = sum(bool(r.get("localized")) for r in lib)
nbug = len({r["case"] for r in lib if r["outcome"] == "REPAIRED"})
ncap = sum("token cap" in " ".join(r.get("reasons", [])) for r in lib)
print("library repair:", nrep, "of", len(lib), "runs;", nbug, "bugs;", nloc, "localized;", sum(r["outcome"] == "OVERFIT" for r in lib), "overfit;", ncap, "ended on the token cap")
check("README", f"{nrep} of {len(lib)} runs (2 of the 7 bugs)" if nbug == 2 else "?", "library repair headline")
check("DEVPOST", f"**{nrep} of {len(lib)} runs repaired, {nbug} of the 7 bugs**", "library repair in Devpost")
check("DEVPOST", f"{nloc} of {len(lib)} runs edited the same file", "localisation")
check("VIDEO", "three of fourteen runs" if (nrep, len(lib)) == (3, 14) else "?", "video wording")
n_llm = sum(1 for q in (ROOT / "results/real_repair/real3/logs").glob("*.jsonl") for l in q.read_text(encoding="utf-8").splitlines() if '"event": "llm_response"' in l)
check("FEEDBACK", f"{n_llm} of {n_llm} model calls in the final run", "tool calls in long loops")
check("README", f"All {ncap} failures ended on the 300k-token cap" if ncap == len(lib) - nrep else "?", "failures ended on the cap")

print("\nMISMATCHES:" if bad else "\nall checked numbers match", bad or "")
raise SystemExit(1 if bad else 0)
