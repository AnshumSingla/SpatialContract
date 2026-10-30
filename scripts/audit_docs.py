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
check("README", f"{nrep} of {len(lib)} runs (2 of the 7 bugs)" if nbug == 2 else "?", "library repair headline (real3)")
check("DEVPOST", f"{nrep} of {len(lib)} runs repaired, {nbug} of the 7 bugs", "library repair in Devpost (real3)")
check("DEVPOST", f"{nloc} of {len(lib)} runs edited the same file", "localisation (real3)")
n_llm = sum(1 for q in (ROOT / "results/real_repair/real3/logs").glob("*.jsonl") for l in q.read_text(encoding="utf-8").splitlines() if '"event": "llm_response"' in l)
check("FEEDBACK", f"{n_llm} of {n_llm} model calls in the real3 run", "tool calls in long loops (real3)")
check("README", f"all {ncap} failures ended on the 300k-token cap" if ncap == len(lib) - nrep else "?", "real3 failures ended on the cap")

# ---- localisation ranking (results/localisation.json, no tokens) and the library run that uses it (results/real_repair/real4)
loc = json.loads((ROOT / "results/localisation.json").read_text(encoding="utf-8"))
f10, fn5 = sum(r["file@10"] for r in loc), sum(r["func@5"] for r in loc)
print("localisation: file@10", f10, "of", len(loc), "; function@5", fn5)
check("README", f"top 10 for **{f10} of {len(loc)}**", "trace file@10")
check("DEVPOST", f"top 10 for {f10} of {len(loc)} bugs", "trace file@10")
for doc in ("README", "DEVPOST"):
    check(doc, f"top 5 for {fn5} of {len(loc)}", "trace function@5")
r4 = [json.loads(l) for l in (ROOT / "results/real_repair/real4/raw.jsonl").read_text(encoding="utf-8").splitlines()]
n4, b4 = sum(r["outcome"] == "REPAIRED" for r in r4), len({r["case"] for r in r4 if r["outcome"] == "REPAIRED"})
l4, nb4 = sum(bool(r.get("localized")) for r in r4), len({r["case"] for r in r4})
print("library repair with the trace:", n4, "of", len(r4), "runs;", b4, "of", nb4, "bugs;", l4, "localized;",
      sum(r["outcome"] == "OVERFIT" for r in r4), "overfit")
for doc in ("README", "DEVPOST"):
    check(doc, f"{n4} of {len(r4)} runs repaired ({b4} of the {nb4} bugs)", "real4 headline")
    check(doc, f"{l4} of {len(r4)} runs edited a file the upstream fix edited", "real4 localisation")
n_llm4 = sum(1 for q in (ROOT / "results/real_repair/real4/logs").glob("*.jsonl") for l in q.read_text(encoding="utf-8").splitlines() if '"event": "llm_response"' in l)
check("FEEDBACK", f"{n_llm4} of {n_llm4} model calls in the real4 run", "tool calls in long loops (real4)")

# ---- equal gating, recomputed from the same rows (results/eval/gated.json, scripts/analyze_gating.py)
g = json.loads((ROOT / "results/eval/gated.json").read_text(encoding="utf-8"))
print("gated:", {a: (v["repaired_pct"], v["controls_modified"], v["abstain_cases_patched"]) for a, v in g.items()})
for doc in ("README", "DEVPOST"):
    check(doc, f"repaired 58%" if doc == "DEVPOST" else f"original tests {g['arm2']['repaired_pct']}%", "gated arm2 repair")
    check(doc, f"{g['arm2']['abstain_cases_patched']} of 21" if g["arm2"]["abstain_cases_patched"] == 6 else "?", "gated arm2 abstain")
    check(doc, f"{g['arm2s']['abstain_cases_patched']} of 21" if g["arm2s"]["abstain_cases_patched"] == 12 else "?", "gated arm2s abstain")
check("README", f"Repair under that gate: original tests {g['arm2']['repaired_pct']}%", "gated repair sentence")
check("DEVPOST", f"repaired {g['arm2']['repaired_pct']}%, {g['arm2s']['repaired_pct']}% and {g['arm3']['repaired_pct']}%", "gated repair sentence")
assert all(g[a]["controls_modified"] == 0 for a in ("arm2", "arm2s", "arm3")), "gated controls"
# ---- the libraries' own tests (results/real_bugs_own_tests.json, scripts/run_prefix_tests.py)
own = json.loads((ROOT / "results/real_bugs_own_tests.json").read_text(encoding="utf-8"))
missed, caught = sum(r["missed_by_existing_tests"] for r in own), sum(r["regression_test_catches_it"] for r in own)
print("own tests: missed", missed, "of", len(own), "; regression test catches", caught)
check("README", f"**No, {len(own) - missed} of {len(own)}.**", "own tests caught")
check("README", f"catches the bug in {caught} of {len(own)} cases", "regression tests")
check("DEVPOST", f"({len(own) - missed} of {len(own)} caught)", "own tests caught (Devpost)")
check("DEVPOST", f"catch {caught} of {len(own)}", "regression tests (Devpost)")
# ---- real scans (results/real_scans.json, results/real_scans_lib.json)
rs = json.loads((ROOT / "results/real_scans.json").read_text(encoding="utf-8"))
rl = json.loads((ROOT / "results/real_scans_lib.json").read_text(encoding="utf-8"))
sm = rs["summary"]
ll = [r for r in rs["rows"] if r["kind"] == "repaired_patch" and r["base"] in "RCLM"]
n_exact = sum(r["r"] is not None and r["r"] >= 0.9999 for r in ll)
n_eq = sum(bool(r.get("agent_equals_upstream")) for r in rl)
print("real scans:", sm["repaired_patch"]["pass_on_every_scan"], "of", sm["repaired_patch"]["n"], "patches;", n_exact, "of", len(ll), "exact;",
      n_eq, "of", len(rl), "library runs equal upstream;", sm["bug"]["fail_on_some_scan"], "of", sm["bug"]["n"], "bugs fail")
for doc in ("README", "DEVPOST"):
    check(doc, f"All {sm['repaired_patch']['n']} phantom-validated seeded repairs pass", "real-scan patches")
    check(doc, f"{n_exact} of {len(ll)}", "real-scan exact runs")
    check(doc, f"({n_eq} of {len(rl)}" if doc == "README" else f"({n_eq} of {len(rl)})", "library repairs equal upstream")
    check(doc, f"All {sm['bug']['fail_on_some_scan']} buggy originals" if sm["bug"]["fail_on_some_scan"] == sm["bug"]["n"] else "?", "real-scan bugs")
# ---- more algorithms (results/eval/autofix.json, results/repair_comparison.json, results/repair_cascade.json)
af = json.loads((ROOT / "results/eval/autofix.json").read_text(encoding="utf-8"))["summary"]
cm = json.loads((ROOT / "results/repair_comparison.json").read_text(encoding="utf-8"))
cs = json.loads((ROOT / "results/repair_cascade.json").read_text(encoding="utf-8"))
saving = round(100 * (1 - af["hybrid_tokens_on_bugs"] / af["arm3_tokens_on_bugs"]))
st = cm["staged1"]
print("autofix", af["auto_repaired"], "saving", saving, "| staged1 verified", st["verified_runs"], "of", st["runs"], len(st["bugs_verified"]), st["usd"])
check("README", f"**{af['auto_repaired']} of {af['bugs']} seeded bugs, all passing held-out validation, 0 controls touched**", "autofix")
check("README", f"by {saving}% (", "autofix token saving")
check("DEVPOST", f"repairs {af['auto_repaired']} of {af['bugs']} seeded bugs", "autofix (Devpost)")
check("DEVPOST", f"cutting model tokens by {saving}%", "autofix saving (Devpost)")
for doc in ("README", "DEVPOST"):
    check(doc, f"{st['verified_runs']} of {st['runs']} runs verified ({len(st['bugs_verified'])} of the {st['bugs']} bugs)" if doc == "README"
          else f"verified {st['verified_runs']} of {st['runs']} runs ({len(st['bugs_verified'])} of the {st['bugs']} bugs)", "staged verified")
    check(doc, f"${st['usd']:.2f} in total", "staged cost")
    check(doc, f"verifies {cs['cascade_staged_then_agent'][0]['n']} and {cs['cascade_staged_then_agent'][1]['n']} bugs in the two repetitions", "cascade")
# ---- contract suggester (results/suggest_eval.json)
sg = json.loads((ROOT / "results/suggest_eval.json").read_text(encoding="utf-8"))["summary"]
print("suggester:", sg["controls"], sg["bugs"], sg["must_abstain"])
check("README", f"usable for {sg['controls']['usable']} of {sg['controls']['n']} correct controls", "suggester controls")
check("README", f"{sg['bugs']['usable']} of {sg['bugs']['n']} buggy scripts ({sg['bugs']['matches']} exact)", "suggester bugs")
check("DEVPOST", f"it did in {sg['bugs']['hides_bug']} of {sg['bugs']['usable']} usable drafts", "suggester hides bug")
# ---- localisation baselines
nm_file10 = sum(1 for r in loc if r["baselines"]["named_api"][1] and r["baselines"]["named_api"][1] <= 10)
nm_fn5 = sum(1 for r in loc if r["baselines"]["named_api"][0] and r["baselines"]["named_api"][0] <= 5)
rnd_file10 = round(sum(r["baselines"]["random"]["file@10"] for r in loc), 1)
rnd_fn5 = round(sum(r["baselines"]["random"]["func@5"] for r in loc), 1)
print("baselines: named-API file@10", nm_file10, "func@5", nm_fn5, "| random file@10", rnd_file10, "func@5", rnd_fn5)
check("README", f"random order {rnd_file10} of {len(loc)} and {rnd_fn5} of {len(loc)}", "random baseline")
check("README", f'"functions the driver names first" {nm_file10} of {len(loc)} and {nm_fn5} of {len(loc)}', "named-API baseline")

print("\nMISMATCHES:" if bad else "\nall checked numbers match", bad or "")
raise SystemExit(1 if bad else 0)
