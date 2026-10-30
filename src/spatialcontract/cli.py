"""Command line:  spatialcontract check FILE[::FUNC] ... [--seeds 0 1 2] [--tol-mm 1.5] [--pythonpath DIR] [--json] [--report F]

Each FILE is checked as
  a script      that declares SPATIAL_CONTRACT and/or defines process(in_path, out_path)   -> oracle.check on the file
  a module      with top-level functions decorated @spatial_contract(...)                   -> api check of each function
The decorator's arguments are read statically (ast.literal_eval), like SPATIAL_CONTRACT, so nothing runs in this process;
the code under test runs only in the oracle's scrubbed subprocess. ``FILE::FUNC`` checks one decorated function.

Exit code: 0 all PASS, 1 any FAIL, 3 any ERROR (and no FAIL), 4 any ABSTAIN (and no FAIL/ERROR); 2 is a usage error.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from . import api, oracle
from .contract import load_from_source

EXIT = {"PASS": 0, "FAIL": 1, "ERROR": 3, "ABSTAIN": 4}
_PRECEDENCE = ("FAIL", "ERROR", "ABSTAIN", "PASS")


def _decorated(tree: ast.Module) -> dict:
    """name -> Declaration for top-level functions decorated with ``spatial_contract`` (bare or called)."""
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            f = call.func if call else dec
            if (f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None) != "spatial_contract":
                continue
            kw = {}
            for k in (call.keywords if call else []):
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except (ValueError, SyntaxError):
                    kw[k.arg] = None                     # not a literal -> field missing -> ABSTAIN
            sig = kw.get("signature", "image")
            out[node.name] = api.declare(**{k: v for k, v in kw.items() if k in api.CONTRACT_FIELDS},
                                         signature=sig if sig in api.SIGNATURES else "image")
    return out


def _targets(arg: str):
    """(label, thunk) pairs for one command-line argument; the thunk takes the run options and returns a Report."""
    path_s, _, func = arg.partition("::")
    path = Path(path_s)
    if not path.is_file():
        raise SystemExit(f"spatialcontract: no such file: {path_s}")
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        reason = f"syntax error: {e}"
        return [(arg, lambda **kw: oracle.Report("ERROR", None, [reason]))]
    decorated = _decorated(tree)
    is_script = (load_from_source(src) is not None
                 or any(isinstance(n, ast.FunctionDef) and n.name == "process" for n in tree.body))
    target = lambda name: {"file": str(path.resolve()), "module": f"sc_target_{path.stem}", "qualname": name}  # noqa: E731
    if func:
        if func not in decorated:
            raise SystemExit(f"spatialcontract: {path_s} has no top-level @spatial_contract function {func!r}")
        decorated, is_script = {func: decorated[func]}, False
    out = []
    if is_script:
        out.append((path_s, lambda seeds, tol_mm, pythonpath: oracle.check(path, None, seeds=seeds, tol_mm=tol_mm,
                                                                           pythonpath=pythonpath or None)))
    for name, d in decorated.items():
        out.append((f"{path_s}::{name}", lambda seeds, tol_mm, pythonpath, t=target(name), d=d:
                    api._check_target(t, d, seeds, tol_mm, pythonpath=pythonpath)))
    if not out:
        reason = "no SPATIAL_CONTRACT, no process(in_path, out_path) and no @spatial_contract function found"
        out.append((path_s, lambda **kw: oracle.Report("ABSTAIN", None, [reason])))
    return out


def _init(a) -> int:
    """Print a suggested contract and its evidence; never edits the file. Exit 0 if every required field was settled, 4 otherwise."""
    from . import suggest
    from .llm import Client, TokenBudget, require_api_enabled
    require_api_enabled()
    base = TokenBudget()
    s = suggest.suggest(Path(a.file).read_text(encoding="utf-8"), Client(budget=TokenBudget(cap=min(base.cap, base.total + a.budget))))
    print(s.block())
    weak = [f for f in suggest.FIELDS if s.basis.get(f) == "inferred"]
    if weak:
        print(f"# inferred from what the code does, not from stated intent (a bug would be copied): {', '.join(weak)}")
    if s.problems:
        print("# not usable yet: " + "; ".join(s.problems) + ". Fill these in yourself; the oracle abstains until you do.")
    return 0 if not s.problems else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spatialcontract", description="Deterministic spatial-contract oracle for imaging code.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run the landmark-phantom oracle on scripts or @spatial_contract functions")
    c.add_argument("files", nargs="+", metavar="FILE[::FUNC]")
    c.add_argument("--seeds", nargs="+", type=int, default=list(oracle.DEFAULT_SEEDS), help="phantom seeds (default 0 1 2)")
    c.add_argument("--tol-mm", type=float, default=oracle.DEFAULT_TOL_MM)
    c.add_argument("--pythonpath", action="append", default=[], metavar="DIR", help="extra import root for the code (repeatable)")
    c.add_argument("--json", action="store_true", help="print machine-readable reports instead of text")
    c.add_argument("--report", metavar="FILE", help="also write the machine-readable reports to FILE (JSON)")
    i = sub.add_parser("init", help="suggest a SPATIAL_CONTRACT for a script with quoted evidence (calls Nemotron; prints only)")
    i.add_argument("file")
    i.add_argument("--budget", type=int, default=20000, help="max tokens this invocation may add to the ledger")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        return _init(a)

    opts = dict(seeds=tuple(a.seeds), tol_mm=a.tol_mm, pythonpath=[str(Path(p).resolve()) for p in a.pythonpath])
    results = []
    for arg in a.files:
        for label, run in _targets(arg):
            rep = run(**opts)
            results.append({"target": label, "verdict": rep.verdict, "summary": api.summary(rep), "report": rep})
            if not a.json:
                print(f"{api.summary(rep)}  [{label}]")
                if rep.verdict in ("FAIL", "ERROR"):
                    print("    " + rep.feedback().replace("\n", "\n    "))
    verdicts = {r["verdict"] for r in results}
    worst = next(v for v in _PRECEDENCE if v in verdicts or v == "PASS")
    rows = json.dumps([r | {"report": r["report"].to_dict()} for r in results], indent=2, default=str)
    if a.report:
        Path(a.report).write_text(rows, encoding="utf-8")
    if a.json:
        print(rows)
    elif worst == "ABSTAIN":
        print("ABSTAIN: no verdict for at least one target; declare a complete spatial contract "
              "(world_frame, units, affine_convention) so the oracle can check it.")
    return EXIT[worst]


if __name__ == "__main__":
    sys.exit(main())
