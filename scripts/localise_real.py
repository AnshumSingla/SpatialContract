"""Measure oracle-guided localisation on the real-bug corpus, with no LLM and no tokens, against simple baselines.

  python scripts/localise_real.py            # all cases -> results/localisation.json
For each case: trace the driver on the pre-fix library over the visible bank plus locate.EXTRA_SEEDS, rank the library
functions, and report where the functions edited by the upstream fix land in that ranking (hit@k). The same executed functions
are also ranked by three baselines: random order (expected over 2000 shuffles), "the driver names it" (functions whose name
appears as an identifier in the driver script first, then everything else, each group in execution order), and the spectrum
(Ochiai) score alone."""
import ast
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import libagent, locate, realbugs  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402

KS = (1, 3, 5, 10)


def first_hits(keys, truth):
    files = {f for f, _ in truth}
    fn = next((i + 1 for i, k in enumerate(keys) if k in set(truth)), None)
    fl = next((i + 1 for i, k in enumerate(keys) if k[0] in files), None)
    return fn, fl


def baselines(sus, truth, driver_src) -> dict:
    names = {n.id for n in ast.walk(ast.parse(driver_src)) if isinstance(n, ast.Name)} |             {n.attr for n in ast.walk(ast.parse(driver_src)) if isinstance(n, ast.Attribute)}
    order = sorted(sus, key=lambda s: s.first_seen)
    named = [s for s in order if set(s.qualname.split(".")) & names]
    out = {"named_api": first_hits([s.key for s in named + [s for s in order if s not in named]], truth),
           "spectrum_only": first_hits([s.key for s in sorted(sus, key=lambda s: (-s.spectrum, s.calls, s.first_seen))], truth)}
    rng, keys, rand = random.Random(0), [s.key for s in sus], []
    for _ in range(2000):
        rng.shuffle(keys)
        rand.append(first_hits(keys, truth))
    out["random"] = {f"{lvl}@{k}": sum(1 for r in rand if r[j] and r[j] <= k) / len(rand) for k in KS for j, lvl in ((0, "func"), (1, "file"))}
    return out


def run_case(case) -> dict:
    ws = libagent.Workspace(case)
    try:
        driver = case.script.read_text(encoding="utf-8")
        loc = ws.locate(load_from_source(driver))
        truth = [tuple(t) for t in ws.truth_functions()]
        return {"case": case.id, "truth": truth, **locate.hits(loc, truth), "statuses": loc.statuses,
                "baselines": baselines(loc.suspects, truth, driver),
                "top": [[s.file, s.line, s.qualname, round(s.score, 3)] for s in loc.suspects[:10]], "summary": loc.summary()}
    finally:
        ws.cleanup()


def main() -> int:
    ids = sys.argv[1:]
    rows = [run_case(c) for c in realbugs.CASES if not ids or c.id in ids]
    for r in rows:
        b = r["baselines"]
        print(f"{r['case']:24s} fail/pass={r['n_fail']}/{r['n_pass']} of {r['n_ranked']}: trace function/file rank "
              f"{r['rank_first_true_function']}/{r['rank_first_true_file']} | named-API {b['named_api'][0]}/{b['named_api'][1]} | "
              f"spectrum-only {b['spectrum_only'][0]}/{b['spectrum_only'][1]}")
        if ids:                                    # a named case: show the ranked list itself
            truth = {tuple(t) for t in r["truth"]}
            for i, (f, line, q, score) in enumerate(r["top"][:8], 1):
                mark = "   <- edited by the upstream fix" if (f, q) in truth else ""
                print(f"  {i}. {f}:{line} {q}  (score {score}){mark}")
    n = len(rows)
    for k in KS:
        hit = lambda rank: rank is not None and rank <= k  # noqa: E731
        print(f"@{k:<2d} function: trace {sum(r[f'func@{k}'] for r in rows)}/{n}, named-API {sum(hit(r['baselines']['named_api'][0]) for r in rows)}/{n}, "
              f"spectrum-only {sum(hit(r['baselines']['spectrum_only'][0]) for r in rows)}/{n}, random {sum(r['baselines']['random'][f'func@{k}'] for r in rows):.1f}/{n} | "
              f"file: trace {sum(r[f'file@{k}'] for r in rows)}/{n}, named-API {sum(hit(r['baselines']['named_api'][1]) for r in rows)}/{n}, "
              f"spectrum-only {sum(hit(r['baselines']['spectrum_only'][1]) for r in rows)}/{n}, random {sum(r['baselines']['random'][f'file@{k}'] for r in rows):.1f}/{n}")
    if not ids:
        (ROOT / "results" / "localisation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
