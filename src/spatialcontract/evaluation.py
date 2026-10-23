"""Three-arm evaluation on the seeded corpus.

arm1  oracle only: generic single-mutation search guided by the oracle, no language model        (mutate.py)
arm2  LLM + repo tests only: same model/loop/guards, feedback = ordinary unit tests, no gate,    (agent.py arm="tests")
      plus report_no_bug / abstain tools
arm3  LLM + oracle feedback: the repair agent with the contract gate and oracle measurements      (agent.py arm="oracle")

Every final patch is scored identically: visible oracle (phantoms 0-2) and then held-out validation (hidden bank, per-run seed
generated after the patch, paired orientation, non-phantom differential). "Repaired" = both pass.
Raw rows are appended to a JSONL file as they finish, so an interrupted run resumes without repeating work or spending tokens twice.
"""
from __future__ import annotations

import ast
import json
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import agent, heldout, mutate, oracle, repo_tests
from .contract import load_from_source
from .seeded import CORPUS

_lock = threading.Lock()


def load_manifest(corpus: Path = CORPUS) -> list[dict]:
    return json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))


def _same_code(a: str, b: str) -> bool:
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return a == b


def score(original: str, final: str | None, run_seed: int) -> dict:
    """Score a final script against the contract declared in the ORIGINAL. Cases without a usable contract cannot be scored."""
    contract = load_from_source(original)
    if contract is None or contract.problems():
        return {"scorable": False, "changed": final is not None and not _same_code(original, final)}
    changed = final is not None and not _same_code(original, final)
    src = final if changed else original
    with tempfile.TemporaryDirectory(prefix="sc_score_") as d:
        p = Path(d) / "final.py"
        p.write_text(src, encoding="utf-8")
        vis = oracle.check(p, contract)
        out = {"scorable": True, "changed": changed, "visible_pass": vis.passed, "heldout_pass": None}
        if vis.passed:
            held = heldout.validate(p, contract, run_seed=run_seed)
            out["heldout_pass"] = held["passed"]
            out["heldout_summary"] = held["summary"]
    out["repaired"] = bool(out["visible_pass"] and out["heldout_pass"])
    out["overfit"] = bool(out["visible_pass"] and out["heldout_pass"] is False)
    return out


def _base_row(arm: str, rep: int, m: dict) -> dict:
    return {"arm": arm, "rep": rep, "id": m["id"], "kind": m["kind"], "expect": m["expect"], "base": m["base"],
            "bug_class": m["bug_class"]}


def run_task(arm: str, rep: int, m: dict, client, *, corpus: Path, log_dir: Path, max_iters: int, max_tokens: int,
             llm_kwargs: dict | None = None, workers: int = 8) -> dict:
    row = _base_row(arm, rep, m)
    path = corpus / m["file"]
    original = path.read_text(encoding="utf-8")
    t = time.time()
    if arm == "arm1":
        r = mutate.search(path, workers=workers, run_seed=rep)
        row.update(outcome=r.outcome, abstained=r.outcome == "ABSTAIN", claimed_no_bug=False, oracle_evaluations=r.evaluations,
                   candidates=r.candidates, mutation=r.mutation, tokens=0, llm_calls=0, iterations=0)
        final = r.final_source if r.outcome in ("REPAIRED", "OVERFIT") else None
        sc = (score(original, final, rep) if final is not None else
              {"scorable": r.outcome != "ABSTAIN", "changed": False, "visible_pass": r.outcome == "NO_CHANGE_NEEDED",
               "heldout_pass": None, "repaired": False, "overfit": False})
        if r.outcome in ("REPAIRED", "OVERFIT"):
            sc["repaired"], sc["overfit"] = r.outcome == "REPAIRED", r.outcome == "OVERFIT"
        row.update(sc)
    else:
        if arm == "arm3":
            r = agent.repair(path, client, max_iters=max_iters, run_seed=rep, max_tokens=max_tokens, log_dir=log_dir / f"{arm}_r{rep}",
                             llm_kwargs=llm_kwargs, arm="oracle")
        else:
            def factory(_contract, base=m["base"], spec=m.get("spec") or {}):
                def evaluate(code_path):
                    res = repo_tests.run_repo_tests(code_path, base, spec)
                    return res.passed, res.text(), "PASS" if res.passed else "FAIL"
                return evaluate
            r = agent.repair(path, client, max_iters=max_iters, run_seed=rep, max_tokens=max_tokens, log_dir=log_dir / f"{arm}_r{rep}",
                             llm_kwargs=llm_kwargs, evaluate_factory=factory, gate=False, validate=False, arm="tests")
        row.update(outcome=r.outcome, abstained=r.outcome == "ABSTAIN", claimed_no_bug=r.outcome == "NO_CHANGE_CLAIMED",
                   tokens=r.tokens["prompt"] + r.tokens["completion"], llm_calls=r.llm_calls, iterations=r.iterations,
                   oracle_evaluations=0)
        if r.outcome == "ABSTAIN":
            sc = {"scorable": False, "changed": False, "repaired": False, "overfit": False}
        elif r.outcome in ("NO_CHANGE_NEEDED", "NO_CHANGE_CLAIMED"):
            sc = score(original, None, rep)
        elif r.outcome == "REPAIRED":      # arm 3 already ran the identical held-out validation
            sc = {"scorable": True, "changed": True, "visible_pass": True, "heldout_pass": True, "repaired": True, "overfit": False}
        elif r.outcome == "OVERFIT":
            sc = {"scorable": True, "changed": True, "visible_pass": True, "heldout_pass": False, "repaired": False, "overfit": True}
        elif r.outcome in ("FAILED", "ACCEPTED_BY_TESTS"):
            sc = score(original, r.final_source, rep)
        else:                              # BUDGET / ERROR
            sc = {"scorable": False, "changed": False, "repaired": False, "overfit": False}
        row.update(sc)
    row["seconds"] = round(time.time() - t, 1)
    return row


def run_evaluation(arms, reps: int, client, out_dir: Path, *, corpus: Path = CORPUS, kinds=("bug", "control", "abstain"),
                   only_ids=None, max_iters: int = 4, max_tokens: int = 8000, llm_kwargs=None, parallel: int = 4,
                   mutate_workers: int = 14) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"
    done = {}
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["arm"], r["rep"], r["id"])] = r
    manifest = [m for m in load_manifest(corpus) if m["kind"] in kinds and (not only_ids or m["id"] in only_ids)]
    tasks = []
    for arm in arms:
        for rep in (range(1) if arm == "arm1" else range(reps)):
            for m in manifest:
                if (arm, rep, m["id"]) not in done:
                    tasks.append((arm, rep, m))

    def work(task):
        arm, rep, m = task
        try:
            row = run_task(arm, rep, m, client, corpus=corpus, log_dir=out_dir / "logs", max_iters=max_iters,
                           max_tokens=max_tokens, llm_kwargs=llm_kwargs, workers=mutate_workers)
        except Exception as e:      # keep the run alive; the failure is recorded, not hidden
            row = _base_row(arm, rep, m) | {"outcome": "HARNESS_ERROR", "error": f"{type(e).__name__}: {str(e)[:300]}", "repaired": False,
                                           "overfit": False, "changed": False, "scorable": False, "tokens": 0}
        with _lock:
            with raw.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"{row['arm']} r{row['rep']} {row['id']:28s} {row['outcome']:18s} repaired={row.get('repaired')} "
                  f"tokens={row.get('tokens')} {row.get('seconds')}s", flush=True)
        return row

    workers = 1 if arms == ["arm1"] or all(a == "arm1" for a in arms) else parallel
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, tasks))
    return [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def per_case(rows: list[dict], arm: str, field: str, ids=None) -> dict[str, float]:
    """Mean over repetitions of a boolean field, per case id."""
    acc: dict[str, list] = {}
    for r in rows:
        if r["arm"] == arm and (ids is None or r["id"] in ids):
            acc.setdefault(r["id"], []).append(float(bool(r.get(field))))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def boot_mean_ci(values: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap CI over cases (cases are the resampling unit; repetitions are averaged within a case)."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(values) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def boot_diff_ci(a: dict[str, float], b: dict[str, float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Paired difference (a - b) over the cases present in both, with a 95% bootstrap CI."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return (float("nan"),) * 3
    d = [a[i] - b[i] for i in ids]
    return boot_mean_ci(d, n_boot, seed)
