"""Model-free repair of a bug inside a library: mutation search over the functions the runtime trace ranks most suspicious.

The script-level search (mutate.py, arm 1) mutates one ``process`` function. Here the targets are the top-k library functions from
locate.py, and the operators are generic, not derived from any upstream fix:
  delete a statement | unwrap (.T, [::-1], transpose, inv, a scalar factor, unary minus, one side of @) | swap the operands of @ |
  numeric constant +1 / -1 / negated / 0 / 1 | add or subtract 1 around an arithmetic expression (off-by-one) | flip + and - |
  flip a comparison (< <=, > >=) | swap the two positional arguments of a call | replace a local by another local |
  dead store: a local whose name contains a spatial word (affine, origin, spacing, direction, ornt, shape) is assigned and never
  read, so assign it to an attribute or subscript of the function that names the same word.
Each candidate is screened on one failing visible phantom, a survivor must pass every visible phantom, and the first survivor (in
suspect rank order, then operator order) goes through the same held-out validation as everything else. No model, no tokens.
"""
from __future__ import annotations

import ast
import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import locate, oracle
from .agent import Log
from .contract import load_from_source
from .libagent import Workspace, fix_files
from .realbugs import RealCase

SPATIAL_WORDS = ("affine", "origin", "spacing", "direction", "ornt", "shape", "zoom")
COMPARE_FLIPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}


@dataclass
class MutResult:
    case: str
    outcome: str                      # REPAIRED | OVERFIT | FAILED | ERROR
    candidates: int = 0
    screened_pass: int = 0
    winner: str | None = None
    touched: list = field(default_factory=list)
    upstream_files: list = field(default_factory=list)
    localized: bool = False
    visible_passed: bool = False
    heldout: dict | None = None
    diff: str = ""
    seconds: float = 0.0
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _find(tree, qualname: str):
    parts = qualname.replace(".<locals>", "").split(".")

    def walk(node, rest):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ch.name == rest[0]:
                if len(rest) == 1:
                    return ch if not isinstance(ch, ast.ClassDef) else None
                found = walk(ch, rest[1:])
                if found:
                    return found
        return None
    return walk(tree, parts)


def mutants(source: str, qualname: str) -> list[tuple[str, str]]:
    """(label, new module source) for every single mutation of function ``qualname`` in ``source``."""
    tree = ast.parse(source)
    fn = _find(tree, qualname)
    if fn is None:
        return []
    nodes = list(ast.walk(fn))
    parents = {c: p for p in nodes for c in ast.iter_child_nodes(p)}
    stores = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    out = []

    def emit(label, idx, make):
        t2 = copy.deepcopy(tree)
        target = list(ast.walk(_find(t2, qualname)))[idx]

        class T(ast.NodeTransformer):
            def visit(self, node):
                if node is target:
                    return make(node)
                return self.generic_visit(node)
        try:
            T().visit(t2)
            ast.fix_missing_locations(t2)
            out.append((label, ast.unparse(t2)))
        except Exception:
            pass

    for i, n in enumerate(nodes):
        p = parents.get(n)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr)) and isinstance(p, (ast.FunctionDef, ast.If, ast.For, ast.With)) \
                and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)):
            emit(f"delete@{n.lineno}", i, lambda old: ast.Pass())
        if isinstance(n, ast.Attribute) and n.attr == "T":
            emit(f"unwrap.T@{n.lineno}", i, lambda old: old.value)
        if isinstance(n, ast.Subscript) and "::-1" in ast.unparse(n.slice):
            emit(f"unwrap[::-1]@{n.lineno}", i, lambda old: old.value)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("transpose", "swapaxes"):
            emit(f"unwrap.{n.func.attr}@{n.lineno}", i, lambda old: old.func.value)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(("linalg.inv", "inv")) and n.args:
            emit(f"unwrap.inv@{n.lineno}", i, lambda old: old.args[0])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            emit(f"unwrap-@{n.lineno}", i, lambda old: old.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            emit(f"matmul-swap@{n.lineno}", i, lambda old: ast.BinOp(old.right, ast.MatMult(), old.left))
            emit(f"matmul-left@{n.lineno}", i, lambda old: old.left)
            emit(f"matmul-right@{n.lineno}", i, lambda old: old.right)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            emit(f"flip+-@{n.lineno}", i, lambda old: ast.BinOp(old.left, ast.Sub() if isinstance(old.op, ast.Add) else ast.Add(), old.right))
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            emit(f"+1@{n.lineno}", i, lambda old: ast.BinOp(old, ast.Add(), ast.Constant(1)))
            emit(f"-1@{n.lineno}", i, lambda old: ast.BinOp(old, ast.Sub(), ast.Constant(1)))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            for k, v in enumerate((n.value + 1, n.value - 1, -n.value, 0, 1)):
                if v != n.value:
                    emit(f"num{k}@{n.lineno}", i, lambda old, v=v: ast.Constant(v))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in COMPARE_FLIPS:
            emit(f"cmp-flip@{n.lineno}", i, lambda old: ast.Compare(old.left, [COMPARE_FLIPS[type(old.ops[0])]()], old.comparators))
        if isinstance(n, ast.Call) and len(n.args) == 2 and not n.keywords:
            emit(f"swap-args@{n.lineno}", i, lambda old: ast.Call(old.func, [old.args[1], old.args[0]], []))
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in stores | params and not (isinstance(p, ast.Call) and p.func is n):
            for other in sorted((stores | params) - {n.id, "self"}):
                if any(w in n.id.lower() for w in SPATIAL_WORDS) and any(w in other.lower() for w in SPATIAL_WORDS):
                    emit(f"{n.id}->{other}@{n.lineno}", i, lambda old, other=other: ast.Name(other, ast.Load()))
    # dead store of a spatial local: assign it where the function reads or writes something naming the same word
    for name in sorted(stores - loads):
        word = next((w for w in SPATIAL_WORDS if w in name.lower()), None)
        if not word:
            continue
        assign = max((n for n in nodes if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)),
                     key=lambda n: n.lineno)
        targets = sorted({ast.unparse(n) for n in nodes if isinstance(n, (ast.Attribute, ast.Subscript))
                          and word in ast.unparse(n).lower() and "(" not in ast.unparse(n)})
        for tgt in targets:
            idx = list(ast.walk(fn)).index(assign)

            def make(old, tgt=tgt, name=name):
                stmt = ast.parse(f"{tgt} = {name}").body[0]
                return [old, stmt]
            emit(f"dead-store:{tgt}={name}@{assign.lineno}", idx, make)
    seen, uniq = set(), []
    for label, src in out:
        if src not in seen and src != ast.unparse(tree):
            seen.add(src)
            uniq.append((label, src))
    return uniq


def repair_mutation(case: RealCase, *, k: int = 6, max_candidates: int = 600, workers: int = 3, log_path: Path | None = None,
                    run_seed: int = 0) -> MutResult:
    contract = load_from_source(case.script.read_text(encoding="utf-8"))
    log = Log(log_path)
    res = MutResult(case=case.id, outcome="FAILED", upstream_files=fix_files(case))
    t0 = time.time()
    spaces = [Workspace(case) for _ in range(workers)]
    try:
        first = spaces[0].oracle(contract)
        failing = [p.seed for p in first.phantoms if p.status != "PASS"] or [first.phantoms[0].seed]
        loc = spaces[0].locate(contract)
        cands = []
        for s in loc.suspects[:k]:
            src_path = spaces[0]._resolve(s.file)
            src = src_path.read_text(encoding="utf-8")
            for label, new in mutants(src, s.qualname):
                cands.append((s.file, s.qualname, label, new))
        cands = cands[:max_candidates]
        res.candidates = len(cands)
        log.event("start", case=case.id, suspects=[[s.file, s.qualname] for s in loc.suspects[:k]], candidates=len(cands))

        def try_one(job):
            idx, (rel, q, label, new) = job
            ws = spaces[idx % workers]
            p = ws._resolve(rel)
            original = p.read_text(encoding="utf-8")
            try:
                compile(new, rel, "exec")
                ws.originals.setdefault(rel, original)
                p.write_text(new, encoding="utf-8")
                ok = ws.oracle(contract, seeds=failing[:1]).passed and ws.oracle(contract).passed
            except SyntaxError:
                ok = False
            finally:
                p.write_text(ws.originals.pop(rel, original), encoding="utf-8")
            return idx, ok

        winner = None
        for start in range(0, len(cands), workers):                 # batches of `workers`, in rank order, first survivor wins
            batch = list(enumerate(cands))[start:start + workers]
            with ThreadPoolExecutor(workers) as ex:
                results = sorted(ex.map(try_one, batch))
            hit = next((i for i, ok in results if ok), None)
            res.screened_pass += sum(ok for _, ok in results)
            if hit is not None:
                winner = cands[hit]
                break
        if winner:
            rel, q, label, new = winner
            ws = spaces[0]
            p = ws._resolve(rel)
            ws.originals.setdefault(rel, p.read_text(encoding="utf-8"))
            p.write_text(new, encoding="utf-8")
            res.visible_passed, res.winner = True, f"{rel}::{q} {label}"
            res.touched, res.diff = ws.touched(), ws.diff()
            held = ws.validate(contract, run_seed=run_seed)
            res.heldout = held
            res.outcome = "REPAIRED" if held["passed"] else "OVERFIT"
        res.localized = bool(set(res.touched) & set(res.upstream_files))
    except Exception as e:
        res.outcome, res.reasons = "ERROR", [f"{type(e).__name__}: {str(e)[:200]}"]
    finally:
        for ws in spaces:
            ws.cleanup()
    res.seconds = round(time.time() - t0, 1)
    log.event("end", outcome=res.outcome, winner=res.winner, candidates=res.candidates, seconds=res.seconds)
    return res
