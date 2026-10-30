"""Did the libraries' OWN tests catch the real bugs? No LLM, no tokens.

For each real-bug case, run the library's test files that cover the fixed code, three ways:
  pre_tests_on_pre    the tests as they were at the pre-fix commit, on the pre-fix library  -> did the existing tests miss the bug?
  post_tests_on_pre   the tests as the fix left them, on the pre-fix library               -> does the regression test catch it?
  post_tests_on_post  the tests as the fix left them, on the fixed library                 -> sanity: they should pass
"Covering tests" are the test files the fix commit edited; where it edited none, the existing test file for the fixed module.
Post-fix test files are written next to the originals in the pre-fix worktree under a temporary name and removed afterwards.

  python scripts/run_prefix_tests.py [case ...]     -> results/real_bugs_own_tests.json
Needs pytest in the two library environments (_work/env_legacy, _work/env_modern)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import realbugs  # noqa: E402
from spatialcontract.libverify import COVERING, names, pytest_run, test_files  # noqa: E402,F401

def with_post_tests(case, which: str, files: list[str]) -> dict:
    """Write the post-fix version of each test file next to the original in the ``which`` worktree, run them, remove them."""
    wt, repo, written = case.worktree(which), realbugs.WORK / "repos" / case.repo, []
    try:
        for f in files:
            src = subprocess.run(["git", "show", f"{case.post}:{f}"], cwd=repo, capture_output=True, text=True, encoding="utf-8")
            if src.returncode:
                continue
            tmp = wt / (f[:-3] + "_scpost.py")
            tmp.write_text(src.stdout, encoding="utf-8")
            written.append(tmp)
        return pytest_run(case, which, [str(p.relative_to(wt)).replace(os.sep, "/") for p in written]) if written else {"status": "NO_TESTS"}
    finally:
        for p in written:
            p.unlink(missing_ok=True)


def run_case(case) -> dict:
    files = test_files(case)
    existing = [f for f in files if (case.worktree("pre") / f).exists()]
    row = {"id": case.id, "repo": case.repo, "env": case.env, "test_files": files, "fix_edited_tests": case.id not in COVERING,
           "pre_tests_on_pre": pytest_run(case, "pre", existing) if existing else {"status": "NO_TESTS", "note": "test file added by the fix"},
           "post_tests_on_pre": with_post_tests(case, "pre", files),
           "post_tests_on_post": pytest_run(case, "post", files)}
    # Old library code on today's numpy/scipy/sklearn fails some tests for reasons unrelated to the bug. A test counts as
    # detecting the bug only if it fails on the pre-fix library and passes on the fixed one with the same test file.
    env = names(row["post_tests_on_post"])
    row["environment_failures"] = sorted(env)
    row["pre_tests_bug_failures"] = sorted(names(row["pre_tests_on_pre"]) - env - names(pytest_run(case, "post", existing)) if existing else set())
    row["catching_tests"] = sorted(names(row["post_tests_on_pre"]) - env)
    row["missed_by_existing_tests"] = not row["pre_tests_bug_failures"]
    row["regression_test_catches_it"] = bool(row["catching_tests"])
    return row


def main() -> int:
    ids = sys.argv[1:]
    rows = []
    for c in realbugs.CASES:
        if ids and c.id not in ids:
            continue
        r = run_case(c)
        rows.append(r)
        fmt = lambda x: f"{x['status']} {x.get('counts', '')}"  # noqa: E731
        print(f"{c.id:24s} files={r['test_files']}\n    pre tests on pre: {fmt(r['pre_tests_on_pre'])}\n"
              f"    post tests on pre: {fmt(r['post_tests_on_pre'])}  failing: {r['post_tests_on_pre'].get('failed', [])[:4]}\n"
              f"    post tests on post: {fmt(r['post_tests_on_post'])}\n    => existing tests missed it: {r['missed_by_existing_tests']}; "
              f"regression test catches it: {r['regression_test_catches_it']} {r['catching_tests'][:3]}", flush=True)
    if not ids:
        (ROOT / "results" / "real_bugs_own_tests.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
