"""Create git worktrees (pre-fix / post-fix) for every real-bug case under _work/wt. Needs network on first run.
Partial clones (history without blobs) go to _work/repos. Does not create the virtualenvs (see README)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract.realbugs import CASES, REPOS, WORK  # noqa: E402


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    (WORK / "repos").mkdir(parents=True, exist_ok=True)
    (WORK / "wt").mkdir(exist_ok=True)
    for name, url in REPOS.items():
        if not (WORK / "repos" / name).exists():
            git("clone", "-q", "--filter=blob:none", url, str(WORK / "repos" / name))
    for c in CASES:
        for which, sha in (("pre", c.pre), ("post", c.post)):
            wt = c.worktree(which)
            if not wt.exists():
                git("worktree", "add", "-q", "--detach", str(wt), sha, cwd=WORK / "repos" / c.repo)
            print(c.id, which, git("rev-parse", "--short", "HEAD", cwd=wt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
