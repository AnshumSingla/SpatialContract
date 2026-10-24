"""Local live demo server (stdlib only). Serves the page and a small JSON API:

  GET  /                 the page, in live mode
  GET  /api/info         {api: bool, budget: int}
  GET  /api/cases        scripts to try (buggy, correct, no contract) with their sources
  POST /api/check        {source} -> oracle report (+ repo-test result for the known cases); runs the script in a sandboxed subprocess
  POST /api/repair       {source} -> {job}; needs --live and SC_ALLOW_API=1; calls Nemotron on Nebius Token Factory
  GET  /api/job/<id>     {running, record}: the same structure the recorded replays use

SECURITY: this executes the submitted Python (in a subprocess with a scrubbed environment and an import allow-list check, which is NOT a
security boundary). It therefore binds to 127.0.0.1 only. Do not expose it.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import agent, demo_data, repo_tests  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402
from spatialcontract.llm import Client, TokenBudget  # noqa: E402

CORPUS = ROOT / "corpus" / "seeded"
MANIFEST = {m["id"]: m for m in json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))}
CASE_IDS = [("L04_right_multiply", "Unit tests pass, landmarks 100+ mm off"), ("R05_stale_affine", "Stale affine after reorientation"),
            ("C02_index_order", "Crop offset in reversed axis order"), ("M03_not_converted", "Metres declared, millimetres written"),
            ("S06_reshape_order", "Fortran-order reshape"), ("K03_S_reference", "Already correct"),
            ("A05_corner_convention", "Ambiguous contract: abstain"), ("A02_no_contract_buggy", "No contract: abstain")]
MAX_SOURCE = 30_000
HTML = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")

STATE = {"api": False, "budget": 0, "client": None, "jobs": {}, "lock": threading.Lock(), "running": None}


def cases() -> list[dict]:
    out = []
    for cid, title in CASE_IDS:
        m = MANIFEST[cid]
        out.append({"id": cid, "title": title, "kind": m["kind"], "base": m["base"], "description": m["description"],
                    "source": (CORPUS / m["file"]).read_text(encoding="utf-8"), "steps": [], "status": "done", "outcome": None})
    return out


def reject_reason(source: str) -> str | None:
    if len(source) > MAX_SOURCE:
        return "script too large for the demo"
    problems = agent.check_patch(source, source)
    problems = [p for p in problems if "contract" not in p.lower()]
    return "; ".join(problems) if problems else None


def run_check(source: str) -> dict:
    why = reject_reason(source)
    if why:
        return {"error": f"Script rejected by the sandbox guard: {why}"}
    contract = load_from_source(source)
    with tempfile.TemporaryDirectory(prefix="sc_demo_") as d:
        p = Path(d) / "script.py"
        p.write_text(source, encoding="utf-8")
        res: dict = {"report": None, "reasons": []}
        if contract is None or contract.problems():
            res["reasons"] = ["no SPATIAL_CONTRACT declared"] if contract is None else contract.problems()
        else:
            res["report"] = demo_data.structured_report(p, contract)
        for cid, _ in CASE_IDS:                       # repo tests are only defined for the shipped pipelines
            m = MANIFEST[cid]
            if (CORPUS / m["file"]).read_text(encoding="utf-8") == source and m["kind"] != "abstain":
                r = repo_tests.run_repo_tests(p, m["base"], m.get("spec") or {})
                res["repo_tests"] = {"passed": r.passed, "text": r.text()}
                break
    return res


def start_job(source: str) -> dict:
    if not STATE["api"]:
        return {"error": "API calls are disabled on this server (start it with --live and SC_ALLOW_API=1)."}
    why = reject_reason(source)
    if why:
        return {"error": f"Script rejected by the sandbox guard: {why}"}
    with STATE["lock"]:
        if STATE["running"]:
            return {"error": "another repair is running; wait for it to finish"}
        job = uuid.uuid4().hex[:8]
        d = Path(tempfile.mkdtemp(prefix=f"sc_job_{job}_"))
        (d / "script.py").write_text(source, encoding="utf-8")
        STATE["jobs"][job] = {"dir": d, "source": source, "result": None, "error": None}
        STATE["running"] = job

    def work():
        try:
            STATE["jobs"][job]["result"] = agent.repair(d / "script.py", STATE["client"], log_dir=d, max_iters=4, max_tokens=8000, arm="oracle")
        except Exception as e:      # shown to the user, not swallowed
            STATE["jobs"][job]["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            STATE["running"] = None

    threading.Thread(target=work, daemon=True).start()
    return {"job": job}


def job_view(job: str) -> dict:
    j = STATE["jobs"].get(job)
    if not j:
        return {"error": "unknown job"}
    d = j["dir"]
    rec = demo_data.build_record(j["source"], d / "script.jsonl", d / "script_work", "script", title="Live run")
    running = STATE["running"] == job
    rec["model"] = STATE["client"].model if STATE["client"] else None
    if j["result"] is not None:
        rec["outcome"] = j["result"].outcome
        rec["tokens_total"] = j["result"].tokens["prompt"] + j["result"].tokens["completion"]
        rec["status"] = "done"
    if j["error"]:
        rec["outcome"], rec["status"] = "FAILED", "done"
        rec["error"] = j["error"]
    return {"running": running, "record": rec}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = HTML.replace("/*__LIVE__*/false", "true")
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._json({"api": STATE["api"], "budget": STATE["budget"]})
        elif self.path == "/api/cases":
            self._json({"cases": cases()})
        elif self.path.startswith("/api/job/"):
            self._json(job_view(self.path.rsplit("/", 1)[1]))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_SOURCE * 2:
            return self._json({"error": "request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            source = str(body.get("source", ""))
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        if self.path == "/api/check":
            self._json(run_check(source))
        elif self.path == "/api/repair":
            self._json(start_job(source))
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):      # quiet; never log request bodies
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live", action="store_true", help="enable the Repair button (needs SC_ALLOW_API=1)")
    ap.add_argument("--budget", type=int, default=60000, help="max tokens this server session may spend")
    a = ap.parse_args()
    if a.live:
        from spatialcontract.llm import require_api_enabled
        require_api_enabled()
        base = TokenBudget()
        STATE.update(api=True, budget=a.budget, client=Client(budget=TokenBudget(cap=min(base.cap, base.total + a.budget))))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"SpatialContract demo on http://127.0.0.1:{a.port}  (live repair: {'ON, budget %d tokens' % a.budget if a.live else 'off'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
