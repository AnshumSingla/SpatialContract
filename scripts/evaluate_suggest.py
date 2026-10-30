"""How good, and how risky, is the contract suggester? Spends tokens (one call per case): needs SC_ALLOW_API=1 and caps.

On every seeded case the declared SPATIAL_CONTRACT is removed and `suggest.suggest` proposes one from the code alone. Reported:
  usable        the suggestion passes SpatialContract.problems() (the gate would let the oracle run)
  matches       every field equals the declared contract (controls and bugs, which declare one)
  stated        fields whose basis is "stated" (the corpus has no docstrings or comments, so little can be)
  hides bug     for a seeded bug, the oracle PASSES the buggy code under the suggested contract: the suggestion described the bug
                as the intent, which is the reason a person must confirm every suggestion
  must-abstain  the 7 cases whose contract is missing or ambiguous: how often the suggester still claims a usable contract

  SC_ALLOW_API=1 SC_TOKEN_CAP=... SC_USD_CAP=... python scripts/evaluate_suggest.py   -> results/suggest_eval.json"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation as ev, oracle, suggest  # noqa: E402
from spatialcontract.contract import from_dict, load_from_source  # noqa: E402
from spatialcontract.llm import Client, TokenBudget, require_api_enabled  # noqa: E402


def main() -> int:
    require_api_enabled()
    base = TokenBudget()
    client = Client(budget=TokenBudget(cap=min(base.cap, base.total + 400_000)))
    rows = []
    for m in ev.load_manifest():
        path = ev.CORPUS / m["file"]
        src = path.read_text(encoding="utf-8")
        declared = load_from_source(src)
        s = suggest.suggest(suggest.strip_contract(src), client)
        row = {"id": m["id"], "kind": m["kind"], "suggested": s.contract, "basis": s.basis, "usable": not s.problems,
               "declared": declared.to_dict() if declared else None, "tokens": s.tokens}
        if declared and not declared.problems():
            row["matches"] = all(s.contract.get(f) == declared.to_dict().get(f) for f in suggest.FIELDS)
        if m["kind"] == "bug" and not s.problems:
            row["hides_bug"] = oracle.check(path, from_dict(s.contract)).verdict == "PASS"
        rows.append(row)
        print(f"{m['id']:28s} {m['kind']:8s} usable={row['usable']!s:5s} matches={row.get('matches')!s:5s} hides_bug={row.get('hides_bug')!s:5s} "
              f"{s.contract} {s.basis}", flush=True)
    K = lambda k: [r for r in rows if r["kind"] == k]  # noqa: E731
    summary = {
        "controls": {"n": len(K("control")), "usable": sum(r["usable"] for r in K("control")), "matches": sum(bool(r.get("matches")) for r in K("control"))},
        "bugs": {"n": len(K("bug")), "usable": sum(r["usable"] for r in K("bug")), "matches": sum(bool(r.get("matches")) for r in K("bug")),
                 "hides_bug": sum(bool(r.get("hides_bug")) for r in K("bug"))},
        "must_abstain": {"n": len(K("abstain")), "claimed_usable": sum(r["usable"] for r in K("abstain"))},
        "fields_stated": sum(v == "stated" for r in rows for v in r["basis"].values()), "fields": len(rows) * len(suggest.FIELDS),
        "tokens": sum(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rows)}
    (ROOT / "results/suggest_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
