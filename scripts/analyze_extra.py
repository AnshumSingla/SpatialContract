"""Two extra baselines run after the main evaluation, compared with the main arms on the same 49 cases. No tokens, no network.
  python scripts/analyze_extra.py   -> prints markdown, writes results/eval/extra/table.md and summary.json

arm0   same model, no feedback of any kind: it sees the script once and its first patch is final
arm2s  same model + a STRONGER unit-test suite that includes a physical landmark test (written by the author after the first
       evaluation, with the bug classes known; deliberately strong, not independent)
arm2 and arm3 come from results/eval/main (unchanged); arm0 and arm2s from results/eval/extra."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev  # noqa: E402
from spatialcontract import repo_tests  # noqa: E402

ARMS = {"arm0": "0. LLM, no feedback", "arm2": "2. LLM + weak unit tests", "arm2s": "2s. LLM + stronger unit tests", "arm3": "3. LLM + oracle feedback"}


def pct(x):
    return "n/a" if x != x else f"{100 * max(0.0, x):.0f}%"


def fmt(mean, lo, hi):
    return f"{pct(mean)} [{pct(lo)}, {pct(hi)}]"


def load(run):
    return [json.loads(l) for l in (ROOT / "results" / "eval" / run / "raw.jsonl").read_text(encoding="utf-8").splitlines()]


def main() -> None:
    rows = [r for r in load("main") if r["arm"] in ("arm2", "arm3")] + load("extra")
    manifest = {m["id"]: m for m in ev.load_manifest()}
    bugs = {i for i, m in manifest.items() if m["kind"] == "bug"}
    ctrl = {i for i, m in manifest.items() if m["kind"] == "control"}
    abst = {i for i, m in manifest.items() if m["kind"] == "abstain"}
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    out, md = {"arms": arms, "harness_errors": [f"{r['arm']}/{r['id']}" for r in rows if r["outcome"] == "HARNESS_ERROR"]}, []

    flagged_weak = {i for i in bugs if not repo_tests.run_repo_tests(ev.CORPUS / manifest[i]["file"], manifest[i]["base"], manifest[i].get("spec") or {}).passed}
    flagged_strong = {i for i in bugs if not repo_tests.run_repo_tests(ev.CORPUS / manifest[i]["file"], manifest[i]["base"], manifest[i].get("spec") or {}, strong=True).passed}
    out["flagged"] = {"weak": len(flagged_weak), "strong": len(flagged_strong), "bugs": len(bugs)}
    md.append(f"Unit tests flag {len(flagged_weak)} of {len(bugs)} bugs (weak suite) and {len(flagged_strong)} of {len(bugs)} (stronger suite).\n")

    md.append("### Repair on the 31 seeded bugs (mean over repetitions per case; 95% bootstrap CI over cases)\n")
    md.append("| arm | repaired (visible + held-out) | tokens / case | LLM calls / case |")
    md.append("|---|---|---|---|")
    rep = {}
    for a in arms:
        rep[a] = ev.per_case(rows, a, "repaired", bugs)
        ar = [r for r in rows if r["arm"] == a and r["id"] in bugs]
        ci = ev.boot_mean_ci(list(rep[a].values()))
        tok, calls = sum(r.get("tokens", 0) for r in ar) / len(ar), sum(r.get("llm_calls", 0) for r in ar) / len(ar)
        md.append(f"| {ARMS[a]} | {fmt(*ci)} | {tok:.0f} | {calls:.1f} |")
        out.setdefault("repair", {})[a] = {"repaired": ci, "tokens_per_case": tok, "calls_per_case": calls, "n_rows": len(ar)}
    for other in ("arm0", "arm2", "arm2s"):
        if other in rep and "arm3" in rep:
            d = ev.boot_diff_ci(rep["arm3"], rep[other])
            md.append(f"\nPaired difference, arm 3 - {other}: **{100 * d[0]:+.0f} points** (95% CI {100 * d[1]:+.0f} to {100 * d[2]:+.0f}).")
            out.setdefault("diff_vs_arm3", {})[other] = d

    md.append("\n### Split by whether the WEAK unit tests flag the bug at the start\n")
    md.append(f"| arm | weak tests flag it (n={len(flagged_weak)}) | weak tests do not flag it (n={len(bugs) - len(flagged_weak)}) |")
    md.append("|---|---|---|")
    for a in arms:
        f1 = ev.boot_mean_ci([v for k, v in rep[a].items() if k in flagged_weak])
        f0 = ev.boot_mean_ci([v for k, v in rep[a].items() if k not in flagged_weak])
        md.append(f"| {ARMS[a]} | {fmt(*f1)} | {fmt(*f0)} |")

    md.append("\n### Correct controls (11 cases x repetitions): modified correct code, and broke it\n")
    md.append("| arm | modified | broke it |")
    md.append("|---|---|---|")
    for a in arms:
        c = [r for r in rows if r["arm"] == a and r["id"] in ctrl]
        mod = sum(bool(r.get("changed")) for r in c)
        broke = sum(bool(r.get("changed")) and not (r.get("visible_pass") and r.get("heldout_pass")) for r in c)
        md.append(f"| {ARMS[a]} | {mod}/{len(c)} | {broke}/{len(c)} |")
        out.setdefault("controls", {})[a] = {"modified": [mod, len(c)], "broken": [broke, len(c)]}

    md.append("\n### Must-abstain cases (7 x repetitions)\n")
    md.append("| arm | abstained | patched instead |")
    md.append("|---|---|---|")
    for a in arms:
        c = [r for r in rows if r["arm"] == a and r["id"] in abst]
        ab, guessed = sum(bool(r.get("abstained")) for r in c), sum(bool(r.get("changed")) for r in c)
        md.append(f"| {ARMS[a]} | {ab}/{len(c)} | {guessed}/{len(c)} |")
        out.setdefault("abstention", {})[a] = {"abstained": [ab, len(c)], "guessed": [guessed, len(c)]}

    md.append(f"\nHarness errors: {out['harness_errors'] or 'none'}.")
    text = "\n".join(md)
    (ROOT / "results" / "eval" / "extra" / "table.md").write_text(text, encoding="utf-8")
    (ROOT / "results" / "eval" / "extra" / "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
