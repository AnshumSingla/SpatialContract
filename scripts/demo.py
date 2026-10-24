"""One entry point for the demo.

  python scripts/demo.py replay [case_id]     terminal walkthrough of a recorded real run (no API, no tokens)
  python scripts/demo.py list                 list the recorded runs
  python scripts/demo.py serve                local web UI with the recorded runs plus "check any script with the oracle"
  SC_ALLOW_API=1 python scripts/demo.py serve --live --budget 60000
                                              same UI with a working "Repair with Nemotron" button (spends tokens, capped)
The static replay page for a demo URL is docs/index.html (python demo/build_static.py rebuilds it)."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demo" / "recordings.json"


def load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def say(s: str = "") -> None:
    print(s.encode(sys.stdout.encoding or "ascii", "replace").decode(sys.stdout.encoding or "ascii"))


def replay(case_id: str | None, which: int = 0) -> int:
    d = load()
    recs = d["recordings"]
    matches = [r for r in recs if case_id is None or r["id"] == case_id]
    if not matches:
        say("unknown case; try: python scripts/demo.py list")
        return 1
    r = matches[min(which, len(matches) - 1)] if case_id else recs[0]
    say("=" * 78)
    say(f"{r['title']}   [{r['id']}]")
    say(f"model: {d['model']}   (recorded run: {r['source_run']})")
    say("=" * 78)
    say("SPATIAL_CONTRACT: " + json.dumps(r["contract"]))
    rt = r.get("repo_tests")
    if rt:
        say(f"\n1. Repo unit tests on the original script: {'ALL PASS (the bug ships green)' if rt['passed'] else 'FAILING'}")
        for line in rt["text"].splitlines():
            say("     " + line)
    init = r.get("initial")
    if not init:
        say("\n2. Oracle: ABSTAIN. " + "; ".join(r.get("abstain_reasons") or []))
        say("   No model call is made and no convention is guessed.")
    else:
        bad = sum(1 for p in init["phantoms"] for l in p["landmarks"] if not l["ok"])
        tot = sum(len(p["landmarks"]) for p in init["phantoms"])
        say(f"\n2. Oracle on the original script: {init['verdict']}  ({bad}/{tot} landmark measurements off, worst {init['worst_error_mm']} mm)")
        p0 = init["phantoms"][0]
        say(f"   phantom seed {p0['seed']}: input orientation {p0['in_axcodes']} -> output {p0['out_axcodes']}")
        say("   landmark  should be (mm)              code put it (mm)           error")
        for l in p0["landmarks"]:
            e = "LOST" if l["error_mm"] is None else f"{l['error_mm']:.1f} mm"
            say(f"   {l['label']:>5}     {str(l['expected']):<28}  {str(l['got']):<24} {e}")
    if r["steps"]:
        say("\n3. Agent loop")
    for s in r["steps"]:
        o = s.get("oracle") or {}
        w = f", worst {o['worst_error_mm']} mm" if o.get("worst_error_mm") is not None else ""
        say(f"   iteration {s['iteration']}: {s.get('verdict', '')}{w}   ({s['tokens']} tokens)")
        for line in textwrap.wrap(s.get("rationale", ""), 70):
            say("      " + line)
    if r.get("heldout"):
        say("\n4. Checks the model never saw: " + ", ".join(f"{k}={v}" for k, v in r["heldout"]["summary"].items()))
    elif r["steps"]:
        say("\n4. No patch passed the visible oracle, so held-out validation never ran (reported as a failure).")
    say(f"\nOUTCOME: {r['outcome']}   total tokens: {r['tokens_total']}")
    if rt and rt.get("baseline_accepted"):
        bo = rt.get("baseline_oracle") or {}
        say(f"For comparison, the same model with only the repo tests accepted a patch after one call; the oracle then says "
            f"{bo.get('verdict')}, worst landmark {bo.get('worst_error_mm')} mm.")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        say(__doc__)
        return 0
    if a[0] == "list":
        for i, r in enumerate(load()["recordings"]):
            say(f"{r['id']:26s} {r['outcome']:18s} {r['title']}")
        return 0
    if a[0] == "replay":
        return replay(a[1] if len(a) > 1 else None)
    if a[0] == "serve":
        return subprocess.call([sys.executable, str(ROOT / "demo" / "server.py"), *a[1:]])
    say(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
