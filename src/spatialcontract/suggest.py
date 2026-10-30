"""Suggest a SPATIAL_CONTRACT for a pipeline that has none, for a human to confirm. Never applied automatically.

The contract states what the output MUST mean; the code shows what it currently does, which is wrong when there is a bug. So the
model (Nemotron through Token Factory) must ground every field in evidence quoted from the source and say what kind it is:
  stated     a docstring, comment, name or argument that says the intended convention ("returns RAS mm", ``to_lps=True``)
  inferred   only what the code does (a call to as_closest_canonical, a /1000) - a description of the code, possibly of its bug
and answer "unknown" when the source does not settle a field. Values outside the supported ones become "unknown". The result is
printed for review; the gate keeps abstaining until a person puts the contract into the file, so the "never guess" rule holds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .contract import CONVENTIONS, FRAMES, UNIT_TO_MM, from_dict

FIELDS = ("world_frame", "units", "affine_convention", "array_orientation")
_ITEM = {"type": "object", "required": ["value", "basis", "evidence"], "properties": {
    "value": {"type": "string", "description": "the value, or 'unknown' (for array_orientation also 'none' = not required)"},
    "basis": {"type": "string", "enum": ["stated", "inferred", "unknown"],
              "description": "stated: the source says the intended convention; inferred: only what the code does; unknown"},
    "evidence": {"type": "string", "description": "the exact source text (quote it) that supports the value, or '' when unknown"}}}
TOOL = [{"type": "function", "function": {
    "name": "propose_contract",
    "description": "Propose the spatial contract the pipeline's OUTPUT is meant to satisfy, field by field, with quoted evidence.",
    "parameters": {"type": "object", "required": list(FIELDS), "properties": {f: _ITEM for f in FIELDS}}}}]

SYSTEM = """You help a medical-imaging developer write a SPATIAL_CONTRACT: a declaration of what the physical coordinates of a pipeline's OUTPUT
must mean. Fields: world_frame (RAS or LPS: where +x/+y point in patient space), units of the affine's world coordinates (mm, m or um),
affine_convention (only voxel_center_to_world is supported), and array_orientation (optional: a 3-letter code like RAS that the output voxel
axes must follow, or none if the pipeline does not promise one).

The contract is about INTENT. The code may contain a bug, so what the code does is weak evidence of what it should do. For every field:
- quote the exact source text you rely on;
- basis "stated" only if the text states the intended convention (docstring, comment, function or variable name, an argument such as
  to_lps=True or units="m"); basis "inferred" if you only read what the code does; "unknown" if the source does not settle it;
- prefer "unknown" to a guess. Never invent evidence. NIfTI files are conventionally RAS mm, but say "inferred" when that is your only reason.
Reply with one call to propose_contract."""


@dataclass
class Suggestion:
    contract: dict                     # field -> value ("unknown" where not settled), ready to review
    basis: dict                        # field -> stated | inferred | unknown
    evidence: dict                     # field -> quoted source text
    problems: list = field(default_factory=list)       # SpatialContract.problems() of the suggestion
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})

    def block(self) -> str:
        """A SPATIAL_CONTRACT literal to paste, with the evidence as comments."""
        lines = ["SPATIAL_CONTRACT = {   # suggested by `spatialcontract init`: REVIEW before use; the gate abstains until it is in the file"]
        for f in FIELDS:
            v = self.contract.get(f)
            if f == "array_orientation" and v is None:
                continue
            ev = _window(self.evidence.get(f, "").replace("\n", " "), f)
            lines.append(f"    {f!r}: {v!r},   # {self.basis.get(f, 'unknown')}: {ev}" if ev else f"    {f!r}: {v!r},   # {self.basis.get(f, 'unknown')}")
        lines.append("}")
        return "\n".join(lines)


_KEYS = {"world_frame": ("RAS", "LPS", "radiolog", "neurolog"), "units": ("millimet", "metre", "meter", " mm", "micro"),
         "affine_convention": ("centre", "center", "corner"), "array_orientation": ("RAS", "LPS", "canonical", "orient")}


def _window(text: str, f: str, width: int = 90) -> str:
    """The part of a long quote around the word that matters for this field."""
    if len(text) <= width:
        return text
    low = text.lower()
    hit = min((i for i in (low.find(k.lower()) for k in _KEYS.get(f, ())) if i >= 0), default=0)
    start = max(0, hit - width // 3)
    return ("..." if start else "") + text[start:start + width] + "..."


def _clean(f: str, item: dict) -> tuple:
    v = str(item.get("value", "unknown")).strip()
    basis = item.get("basis", "unknown") if item.get("basis") in ("stated", "inferred", "unknown") else "unknown"
    ev = str(item.get("evidence", "")).strip()
    allowed = {"world_frame": FRAMES, "units": tuple(UNIT_TO_MM), "affine_convention": CONVENTIONS}.get(f)
    if f == "array_orientation":
        if v.lower() in ("none", "null", "", "not required"):
            return None, basis, ev
        v = v.upper()
        return (v if len(v) == 3 and set(v) <= set("RLAPSI") else "unknown"), basis, ev
    if v not in (allowed or ()):
        return "unknown", "unknown", ev
    if basis == "unknown":
        return "unknown", "unknown", ev
    return v, basis, ev


def suggest(source: str, client, max_tokens: int = 4000) -> Suggestion:
    resp = client.chat([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Pipeline source:\n```python\n{source}\n```"}],
                       max_tokens=max_tokens, tools=TOOL, tool_choice="auto")
    u = resp.usage
    calls = list(resp.choices[0].message.tool_calls or [])
    try:
        args = json.loads(calls[0].function.arguments) if calls else {}
    except ValueError:
        args = {}
    contract, basis, evidence = {}, {}, {}
    for f in FIELDS:
        item = args.get(f) if isinstance(args.get(f), dict) else {}
        contract[f], basis[f], evidence[f] = _clean(f, item)
    problems = from_dict(contract).problems()
    return Suggestion(contract, basis, evidence, problems,
                      {"prompt": u.prompt_tokens if u else 0, "completion": u.completion_tokens if u else 0})


def strip_contract(source: str) -> str:
    """The source without its SPATIAL_CONTRACT assignment (for evaluating the suggester on the corpus)."""
    import ast
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SPATIAL_CONTRACT" for t in node.targets):
            return "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])
    return source
