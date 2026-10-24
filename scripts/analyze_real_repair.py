"""Summarise results/real_repair/<run-id>/raw.jsonl. No tokens, no network (reads git history in _work/repos for the upstream-diff comparison).
  python scripts/analyze_real_repair.py real3"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev  # noqa: E402
from spatialcontract import libagent, realbugs  # noqa: E402


def changed_lines(diff_text: str) -> set[str]:
    return {l[1:].strip() for l in diff_text.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")) and l[1:].strip()}


def upstream_changed_lines(case) -> set[str]:
    files = libagent.fix_files(case)
    out = subprocess.run(["git", "diff", case.pre, case.post, "--", *[(case.src_subdir + "/" if case.src_subdir else "") + f for f in files]],
                         cwd=realbugs.WORK / "repos" / case.repo, capture_output=True, text=True, check=True).stdout
    return changed_lines(out)


def main(run: str) -> None:
    d = ROOT / "results" / "real_repair" / run
    rows = [json.loads(l) for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = {c.id: c for c in realbugs.CASES}
    md = []
    n = len(rows)
    rep = [r for r in rows if r["outcome"] == "REPAIRED"]
    over = [r for r in rows if r["outcome"] == "OVERFIT"]
    loc = [r for r in rows if r.get("localized")]
    vis = [r for r in rows if r.get("visible_passed")]
    lo, hi = ev.wilson(len(rep), n)
    md.append(f"Runs: {n} ({len({r['case'] for r in rows})} bugs x {max(r['rep'] for r in rows) + 1} repetitions). Model: {rows[0].get('model')}.\n")
    md.append(f"- **Repaired** (visible oracle + held-out validation): {len(rep)}/{n} runs ({100 * len(rep) / n:.0f}%, Wilson 95% {100 * lo:.0f}-{100 * hi:.0f}%)")
    md.append(f"- Passed the visible oracle: {len(vis)}/{n}; of those, failed held-out validation (OVERFIT): {len(over)}")
    md.append(f"- Edited a file the upstream fix also edited (localisation): {len(loc)}/{n}")
    md.append(f"- Bugs repaired at least once: {len({r['case'] for r in rep})}/{len({r['case'] for r in rows})}")
    tok = [r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rows if "tokens" in r]
    md.append(f"- Tokens per run: median {sorted(tok)[len(tok) // 2]:,}; total {sum(tok):,}\n")
    md.append("| bug | rep | outcome | edited files | upstream file edited | patch identical to upstream | oracle runs | steps | tokens |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["case"], r["rep"])):
        up = upstream_changed_lines(cases[r["case"]])
        diff = (d / "logs" / f"{r['case']}_r{r['rep']}.diff")
        agent_lines = changed_lines(diff.read_text(encoding="utf-8")) if diff.exists() else set()
        same = "yes" if agent_lines and agent_lines == up else ("partly" if agent_lines & up else "no")
        t = r.get("tokens", {})
        md.append(f"| {r['case']} | {r['rep']} | {r['outcome']} | {', '.join(Path(f).name for f in r.get('touched', [])) or '-'} | {'yes' if r.get('localized') else 'no'} | "
                  f"{same if agent_lines else '-'} | {r.get('oracle_runs')} | {r.get('steps')} | {t.get('prompt', 0) + t.get('completion', 0):,} |")
    text = "\n".join(md)
    (d / "table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "real3")
