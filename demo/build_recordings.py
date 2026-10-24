"""Build demo/recordings.json from REAL runs already on disk (results/eval/main/logs). No API calls, no tokens.

Each recording is one actual agent run (the run-id and repetition are written into the record). The oracle is re-run on every
saved patch to get the landmark positions the page draws; the model's rationale, token counts and held-out results come from the log.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import demo_data  # noqa: E402

EVAL = ROOT / "results" / "eval" / "main"
CORPUS = ROOT / "corpus" / "seeded"
MANIFEST = {m["id"]: m for m in json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))}
MODEL = "nvidia/nemotron-3-super-120b-a12b"

# (case id, arm3 repetition, headline, include the arm-2 (repo tests only) comparison from the same repetition)
PICKS = [
    ("L04_right_multiply", 0, "Unit tests pass. The landmarks are 100+ mm off.", True),
    ("R05_stale_affine", 1, "Stale affine after reorientation", True),
    ("C02_index_order", 0, "Crop offset applied in reversed axis order", True),
    ("M03_not_converted", 0, "Metres declared, millimetres written", True),
    ("S06_reshape_order", 1, "Fortran-order reshape: repaired on the 3rd try", False),
    ("S06_reshape_order", 0, "The same bug, a failed run (published on purpose)", False),
    ("A05_corner_convention", 0, "Ambiguous contract: the agent abstains", False),
    ("A02_no_contract_buggy", 0, "No contract: it will not guess a convention", False),
    ("K03_S_reference", 0, "Already correct: nothing is touched", False),
]


LIB_PICKS = [
    ("nilearn_reorder", 0, "Real library bug: nilearn reorder_img"),
    ("torchio_sitk_ras_lps", 1, "Real library bug: TorchIO RAS/LPS (a different fix than upstream)"),
    ("torchio_crop", 0, "Real library bug: TorchIO Crop (a failed run)"),
]


def main() -> None:
    raw = [json.loads(l) for l in (EVAL / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    records = []
    for cid, rep, title, with_arm2 in PICKS:
        m = MANIFEST[cid]
        original = (CORPUS / m["file"]).read_text(encoding="utf-8")
        a3 = EVAL / "logs" / f"arm3_r{rep}"
        rec = demo_data.build_record(original, a3 / f"{cid}.jsonl", a3 / f"{cid}_work", cid, title=title)
        row = next(r for r in raw if r["arm"] == "arm3" and r["rep"] == rep and r["id"] == cid)
        rec.update(kind=m["kind"], bug_class=m["bug_class"], description=m["description"], base=m["base"], model=MODEL,
                   source_run=f"results/eval/main/logs/arm3_r{rep}/{cid}.jsonl", outcome=row["outcome"], tokens_total=row["tokens"],
                   llm_calls=row["llm_calls"])
        if m["kind"] == "bug" or with_arm2:
            a2 = EVAL / "logs" / f"arm2_r{rep}"
            rec["repo_tests"] = demo_data.repo_test_block(original, m["base"], m.get("spec"), a2 / f"{cid}.jsonl", a2 / f"{cid}_work", cid)
            a2row = next((r for r in raw if r["arm"] == "arm2" and r["rep"] == rep and r["id"] == cid), None)
            if a2row:
                rec["repo_tests"]["baseline_outcome"] = a2row["outcome"]
                rec["repo_tests"]["baseline_repaired"] = bool(a2row.get("repaired"))
        records.append(rec)
        print(f"{cid:26s} r{rep} outcome={rec['outcome']:18s} steps={len(rec['steps'])} tokens={rec['tokens_total']}"
              f" repo_tests_pass={rec.get('repo_tests', {}).get('passed')}", flush=True)
    # real-library repair runs (results/real_repair/real3): the agent edits a private copy of the library itself
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_real_repair as ar
    from spatialcontract import realbugs
    cases = {c.id: c for c in realbugs.CASES}
    run = ROOT / "results" / "real_repair" / "real3"
    for cid, rep, title in LIB_PICKS:
        rec = demo_data.build_library_record(cases[cid], run, rep, title=title)
        agent_lines = ar.changed_lines(rec["diff"])
        up = ar.upstream_changed_lines(cases[cid])
        rec["same_as_upstream"] = "yes" if agent_lines and agent_lines == up else ("partly" if agent_lines & up else "no")
        records.append(rec)
        fin = rec.get("final") or {}
        print(f"{cid:24s} r{rep} LIBRARY outcome={rec['outcome']:9s} events={len(rec['events'])} initial={rec['initial']['verdict']} final={fin.get('verdict')} "
              f"same_as_upstream={rec['same_as_upstream']}", flush=True)
    out = ROOT / "demo" / "recordings.json"
    out.write_text(json.dumps({"model": MODEL, "recordings": records}, indent=1), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
