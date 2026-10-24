"""Build docs/index.html: the demo page with the recorded runs embedded. One file, no network, no API calls.
Serve docs/ with GitHub Pages (or any static host) to get a demo URL."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build() -> Path:
    html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    data = json.dumps(json.loads((ROOT / "demo" / "recordings.json").read_text(encoding="utf-8")), separators=(",", ":"))
    data = data.replace("</", "<\\/")                     # keep the JSON from closing the script tag
    assert "/*__DATA__*/null" in html
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", data), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print("wrote", p, f"{p.stat().st_size // 1024} KB")
