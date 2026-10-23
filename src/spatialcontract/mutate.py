"""Arm 1, "oracle only": repair by generate-and-test with NO language model.

Candidates are all single AST mutations of ``process`` from a fixed, generic operator set (not derived from the seeded bugs'
list, but note the seeded bugs WERE built from single edits, so this search has a structural advantage on that corpus):
  delete a statement | unwrap (drop .T / [::-1] / inv / transpose / scalar factor / unary minus / one side of @) |
  change a numeric constant (x10, /10, x1e3, x1e-3, negate, 0, 1) | change a string constant to another one used in the file |
  drop a keyword argument | swap the operands of @ | replace a variable by another local | wrap a variable or call in
  .T, np.linalg.inv, a flip on axis 0/1/2, a transpose, or a scale factor.
The oracle on one phantom screens each candidate; a survivor must pass all visible phantoms. The first survivor in the
fixed operator order is taken, then validated on the held-out bank like any other patch.
"""
from __future__ import annotations

import ast
import copy
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import agent, heldout, oracle
from .contract import load_from_source

WRAPPERS = ["np.linalg.inv(__X__)", "__X__.T", "__X__[::-1]", "__X__[:, ::-1]", "__X__[:, :, ::-1]",
            "__X__.transpose(1, 0, 2)", "__X__.transpose(0, 2, 1)", "__X__.transpose(2, 1, 0)",
            "__X__ * 1e-3", "__X__ * 1000.0", "__X__ * 0.1", "__X__ * 10.0", "-__X__"]
NUM_CHANGES = [lambda c: c * 10, lambda c: c / 10, lambda c: c * 1e3, lambda c: c * 1e-3, lambda c: -c, lambda c: 0, lambda c: 1]
MODULE_ALIASES = {"np", "nib", "o", "numpy", "nibabel"}
MAX_CANDIDATES = 2500


def _process_fn(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process")


def _parents(fn):
    return {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}


def _apply(tree, idx, make):
    """Deep-copy ``tree`` and replace the idx-th node (ast.walk order) of ``process`` by ``make(old)``; None deletes a statement."""
    t2 = copy.deepcopy(tree)
    fn2 = _process_fn(t2)
    target = list(ast.walk(fn2))[idx]

    class T(ast.NodeTransformer):
        def visit(self, node):
            if node is target:
                return make(node)
            return self.generic_visit(node)

    T().visit(fn2)
    ast.fix_missing_locations(t2)
    return t2


def _wrap(template):
    def make(old):
        new = ast.parse(template, mode="eval").body

        class S(ast.NodeTransformer):
            def visit_Name(self, node):
                return copy.deepcopy(old) if node.id == "__X__" else node
        return S().visit(new)
    return make


def generate(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    fn = _process_fn(tree)
    nodes = list(ast.walk(fn))
    parents = _parents(fn)
    local_names = {a.arg for a in fn.args.args} | {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    str_consts = sorted({n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)})
    groups: dict[str, list[tuple[str, object]]] = {k: [] for k in ("delete", "unwrap", "const", "matmul", "swap", "wrap")}

    for i, n in enumerate(nodes):
        parent = parents.get(n)
        # --- delete a statement (not the function's last statement, which is the save)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr)) and n in fn.body and n is not fn.body[-1]:
            groups["delete"].append((f"delete@{i}", _apply(tree, i, lambda old: None)))
        # --- unwrap
        if isinstance(n, ast.Attribute) and n.attr == "T":
            groups["unwrap"].append((f"unwrap.T@{i}", _apply(tree, i, lambda old: old.value)))
        elif isinstance(n, ast.Subscript) and isinstance(n.slice, (ast.Slice, ast.Tuple)) and "::-1" in ast.unparse(n.slice):
            groups["unwrap"].append((f"unwrap[::-1]@{i}", _apply(tree, i, lambda old: old.value)))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("transpose", "swapaxes", "T"):
            groups["unwrap"].append((f"unwrap.{n.func.attr}@{i}", _apply(tree, i, lambda old: old.func.value)))
        elif isinstance(n, ast.Call) and ast.unparse(n.func).endswith("linalg.inv") and n.args:
            groups["unwrap"].append((f"unwrap.inv@{i}", _apply(tree, i, lambda old: old.args[0])))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and any(isinstance(s, ast.Constant) for s in (n.left, n.right)):
            keep = n.right if isinstance(n.left, ast.Constant) else n.left
            groups["unwrap"].append((f"unwrap*const@{i}", _apply(tree, i, lambda old, keep=keep: keep)))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            groups["unwrap"].append((f"unwrap-@{i}", _apply(tree, i, lambda old: old.operand)))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            groups["unwrap"].append((f"matmul-left@{i}", _apply(tree, i, lambda old: old.left)))
            groups["unwrap"].append((f"matmul-right@{i}", _apply(tree, i, lambda old: old.right)))
            groups["matmul"].append((f"matmul-swap@{i}", _apply(tree, i, lambda old: ast.BinOp(old.right, ast.MatMult(), old.left))))
        # --- constants
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            for k, f in enumerate(NUM_CHANGES):
                v = f(n.value)
                if v != n.value:
                    groups["const"].append((f"num{k}@{i}", _apply(tree, i, lambda old, v=v: ast.Constant(v))))
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            for s in str_consts:
                if s != n.value:
                    groups["const"].append((f"str->{s}@{i}", _apply(tree, i, lambda old, s=s: ast.Constant(s))))
        if isinstance(n, ast.keyword) and n.arg:
            groups["unwrap"].append((f"drop-kw({n.arg})@{i}", _apply(tree, i, lambda old: None)))
        # --- variable replacement and wrapping (Name loads that are not module aliases or call targets)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in MODULE_ALIASES:
            if not (isinstance(parent, ast.Call) and parent.func is n) and not (isinstance(parent, ast.Attribute) and parent.value is n and False):
                for other in sorted(local_names - {n.id}):
                    groups["swap"].append((f"{n.id}->{other}@{i}", _apply(tree, i, lambda old, other=other: ast.Name(other, ast.Load()))))
                for w in WRAPPERS:
                    groups["wrap"].append((f"wrap[{w}]@{i}", _apply(tree, i, _wrap(w))))
        if isinstance(n, ast.Call) and not isinstance(parent, ast.Expr):
            for w in WRAPPERS:
                groups["wrap"].append((f"wrap[{w}]@{i}", _apply(tree, i, _wrap(w))))

    out, seen = [], {ast.unparse(tree)}
    for g in ("delete", "unwrap", "const", "matmul", "swap", "wrap"):
        for name, t in groups[g]:
            try:
                src = ast.unparse(t)
                compile(src, "<mutant>", "exec")
            except Exception:
                continue
            if src not in seen:
                seen.add(src)
                out.append((name, src))
    return out[:MAX_CANDIDATES]


@dataclass
class SearchResult:
    case: str
    outcome: str                  # ABSTAIN | NO_CHANGE_NEEDED | REPAIRED | OVERFIT | FAILED
    evaluations: int = 0          # oracle screenings a sequential search would have run
    candidates: int = 0
    mutation: str | None = None
    visible_passed: bool = False
    heldout: dict | None = None
    final_source: str | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("final_source")
        return d


def search(case_path, *, workers: int = 14, run_seed: int = 0, batch: int = 42, validate: bool = True) -> SearchResult:
    case_path = Path(case_path)
    original = case_path.read_text(encoding="utf-8")
    res = SearchResult(case_path.stem, "FAILED")
    contract = load_from_source(original)
    if contract is None or contract.problems():
        res.outcome = "ABSTAIN"
        return res
    if oracle.check(case_path, contract).passed:
        res.outcome, res.visible_passed, res.final_source = "NO_CHANGE_NEEDED", True, original
        return res
    cands = generate(original)
    res.candidates = len(cands)
    with tempfile.TemporaryDirectory(prefix="sc_mut_") as d:
        d = Path(d)

        def screen(item):
            i, (name, src) = item
            p = d / f"m{i}.py"
            p.write_text(src, encoding="utf-8")
            if agent.check_patch(original, src):
                return i, False
            return i, oracle.check(p, contract, seeds=(0,)).passed

        for start in range(0, len(cands), batch):
            chunk = list(enumerate(cands))[start:start + batch]
            with ThreadPoolExecutor(workers) as ex:
                results = list(ex.map(screen, chunk))
            for i, ok in results:                      # earliest survivor in the fixed order
                if not ok:
                    continue
                p = d / f"m{i}.py"
                if oracle.check(p, contract).passed:   # all visible phantoms
                    res.evaluations, res.mutation = i + 1, cands[i][0]
                    res.visible_passed, res.final_source = True, cands[i][1]
                    if validate:
                        held = heldout.validate(p, contract, run_seed=run_seed)
                        res.heldout = held
                        res.outcome = "REPAIRED" if held["passed"] else "OVERFIT"
                    else:
                        res.outcome = "REPAIRED"
                    return res
        res.evaluations = len(cands)
    return res
