"""Turn results/eval/<run-id>/raw.jsonl into tables with confidence intervals. No tokens, no network.
  python scripts/analyze_eval.py main     -> prints markdown, writes results/eval/main/summary.json and table.md"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev  # noqa: E402
from spatialcontract.seeded import CLASSES  # noqa: E402

ARMS = {"arm1": "1. oracle only (no LLM)", "arm2": "2. LLM + repo tests", "arm3": "3. LLM + oracle feedback"}


def pct(x):
    x = max(0.0, x) if x == x else x
    return "n/a" if x != x else f"{100 * x:.0f}%"


def fmt(mean, lo, hi):
    return f"{pct(mean)} [{pct(lo)}, {pct(hi)}]"


def main(run: str) -> None:
    d = ROOT / "results" / "eval" / run
    rows = [json.loads(l) for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = {m["id"]: m for m in ev.load_manifest()}
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    bugs = {i for i, m in manifest.items() if m["kind"] == "bug"}
    ctrls = {i for i, m in manifest.items() if m["kind"] == "control"}
    aband = {i for i, m in manifest.items() if m["kind"] == "abstain"}
    out: dict = {"run": run, "n_rows": len(rows), "harness_errors": [r["id"] + "/" + r["arm"] for r in rows if r["outcome"] == "HARNESS_ERROR"]}
    md = []

    # ---- repair on seeded bugs
    md.append("### Repair on the 31 seeded bugs (mean over repetitions per case; 95% bootstrap CI over cases)\n")
    md.append("| arm | repaired (visible + held-out) | passed visible oracle | overfit among visible passes | tokens / case | LLM calls / case | oracle screenings / case |")
    md.append("|---|---|---|---|---|---|---|")
    repaired = {}
    for a in arms:
        rep = ev.per_case(rows, a, "repaired", bugs)
        vis = ev.per_case(rows, a, "visible_pass", bugs)
        repaired[a] = rep
        r_ci, v_ci = ev.boot_mean_ci(list(rep.values())), ev.boot_mean_ci(list(vis.values()))
        arows = [r for r in rows if r["arm"] == a and r["id"] in bugs]
        n_vis = sum(bool(r.get("visible_pass")) for r in arows)
        n_over = sum(bool(r.get("overfit")) for r in arows)
        ov = ev.wilson(n_over, n_vis) if n_vis else (float("nan"),) * 2
        tok = sum(r.get("tokens", 0) for r in arows) / max(1, len(arows))
        calls = sum(r.get("llm_calls", 0) for r in arows) / max(1, len(arows))
        evs = sum(r.get("oracle_evaluations", 0) for r in arows) / max(1, len(arows))
        md.append(f"| {ARMS[a]} | {fmt(*r_ci)} | {fmt(*v_ci)} | {n_over}/{n_vis} ({pct(ov[0])}-{pct(ov[1])}, Wilson) | {tok:.0f} | {calls:.1f} | {evs:.0f} |")
        out.setdefault("repair", {})[a] = {"repaired": r_ci, "visible": v_ci, "overfit": [n_over, n_vis], "tokens_per_case": tok,
                                           "calls_per_case": calls, "screenings_per_case": evs, "n_rows": len(arows)}
    if "arm3" in arms and "arm2" in arms:
        dm = ev.boot_diff_ci(repaired["arm3"], repaired["arm2"])
        md.append(f"\nPaired difference, arm 3 - arm 2, repaired rate: **{100 * dm[0]:+.0f} points** (95% CI {100 * dm[1]:+.0f} to {100 * dm[2]:+.0f}).")
        out["arm3_minus_arm2"] = dm
    if "arm3" in arms and "arm1" in arms:
        dm = ev.boot_diff_ci(repaired["arm3"], repaired["arm1"])
        md.append(f"Paired difference, arm 3 - arm 1: {100 * dm[0]:+.0f} points (95% CI {100 * dm[1]:+.0f} to {100 * dm[2]:+.0f}).")
        out["arm3_minus_arm1"] = dm

    # ---- by whether the repo tests detect the bug at the start
    detected = {}
    try:
        from spatialcontract import repo_tests
        for i in bugs:
            m = manifest[i]
            detected[i] = not repo_tests.run_repo_tests(ev.CORPUS / m["file"], m["base"], m.get("spec") or {}).passed
    except Exception as e:                      # pragma: no cover
        md.append(f"(repo-test detection unavailable: {e})")
    if detected:
        flagged = {i for i, v in detected.items() if v}
        md.append(f"\n### Arm comparison split by whether the repo tests flag the bug at the start ({len(flagged)} flagged / {len(bugs) - len(flagged)} not flagged)\n")
        md.append("| arm | repo tests flag it (n=%d) | repo tests do not flag it (n=%d) |" % (len(flagged), len(bugs) - len(flagged)))
        md.append("|---|---|---|")
        for a in arms:
            f1 = ev.boot_mean_ci([v for k, v in repaired[a].items() if k in flagged])
            f0 = ev.boot_mean_ci([v for k, v in repaired[a].items() if k not in flagged])
            md.append(f"| {ARMS[a]} | {fmt(*f1)} | {fmt(*f0)} |")
            out.setdefault("by_repo_test_detection", {})[a] = {"flagged": f1, "not_flagged": f0}
        out["repo_tests_flag"] = sorted(flagged)

    # ---- by bug class
    md.append("\n### Repaired rate by bug class (case means; small n per class)\n")
    md.append("| class | n | " + " | ".join(ARMS[a] for a in arms) + " |")
    md.append("|---|---|" + "---|" * len(arms))
    for c in CLASSES:
        ids = {i for i in bugs if manifest[i]["bug_class"] == c}
        md.append(f"| {c} | {len(ids)} | " + " | ".join(pct(sum(repaired[a].get(i, 0) for i in ids) / max(1, len(ids))) for a in arms) + " |")

    # ---- controls (false alarms)
    md.append("\n### Correct controls (11): did the arm touch or break code that was already right?\n")
    md.append("| arm | modified a correct pipeline | broke it (fails oracle or held-out afterwards) |")
    md.append("|---|---|---|")
    for a in arms:
        crow = [r for r in rows if r["arm"] == a and r["id"] in ctrls]
        n = len(crow)
        mod = sum(bool(r.get("changed")) for r in crow)
        broke = sum(bool(r.get("changed")) and not (r.get("visible_pass") and r.get("heldout_pass")) for r in crow)
        m_ci, b_ci = ev.wilson(mod, n), ev.wilson(broke, n)
        md.append(f"| {ARMS[a]} | {mod}/{n} ({pct(m_ci[0])}-{pct(m_ci[1])}) | {broke}/{n} ({pct(b_ci[0])}-{pct(b_ci[1])}) |")
        out.setdefault("controls", {})[a] = {"modified": [mod, n], "broken": [broke, n]}
        claim = sum(bool(r.get("claimed_no_bug")) for r in crow)
        if a == "arm2":
            md[-1] += f"  (said 'no bug': {claim}/{n})"

    # ---- abstention
    md.append("\n### Abstention (the 7 must-abstain cases; abstentions elsewhere count against precision)\n")
    md.append("| arm | abstained on must-abstain | patched instead (guessed a convention) | abstained on cases it should have handled | precision | recall |")
    md.append("|---|---|---|---|---|---|")
    for a in arms:
        ab = [r for r in rows if r["arm"] == a and r["id"] in aband]
        other = [r for r in rows if r["arm"] == a and r["id"] not in aband]
        tp = sum(bool(r.get("abstained")) for r in ab)
        guessed = sum(bool(r.get("changed")) for r in ab)
        fp = sum(bool(r.get("abstained")) for r in other)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        md.append(f"| {ARMS[a]} | {tp}/{len(ab)} | {guessed}/{len(ab)} | {fp}/{len(other)} | {pct(prec)} | {pct(tp / len(ab) if ab else float('nan'))} |")
        out.setdefault("abstention", {})[a] = {"tp": tp, "n_must": len(ab), "guessed": guessed, "fp": fp, "n_other": len(other)}
    md.append("\nArms 1 and 3 abstain through the same deterministic gate on the contract, so their abstention numbers measure the gate and are 100% by construction.")

    md.append(f"\nRows: {len(rows)}. Harness errors: {out['harness_errors'] or 'none'}.")
    text = "\n".join(md)
    (d / "table.md").write_text(text, encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
