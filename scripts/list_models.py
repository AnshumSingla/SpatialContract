"""Step 0, zero tokens: GET {base}/models and list Nemotron ids. No completion is made."""
import json
import sys
from pathlib import Path

import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatialcontract.llm import require_api_enabled, settings  # noqa: E402


def main() -> int:
    require_api_enabled()
    s = settings()
    url = s["base_url"].rstrip("/") + "/models?verbose=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {s['api_key']}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status, body = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # network / proxy failures
        print("REQUEST FAILED:", type(e).__name__, str(e)[:200])
        return 1
    print("HTTP", status)
    if status != 200:
        print(body[:300])
        return 1
    items = json.loads(body).get("data", [])
    print("models returned:", len(items))
    out = Path(__file__).resolve().parents[1] / "results" / "models.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(items, indent=1))
    for m in items:
        mid = m.get("id", "")
        if "nemotron" in mid.lower() or "nvidia" in mid.lower():
            keys = {k: v for k, v in m.items() if k not in ("id", "object")}
            print(" ", mid, json.dumps(keys)[:400])
    print("saved full listing to results/models.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
