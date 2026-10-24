"""Build the replay page with the recorded runs embedded. One file, no network, no API calls.

  python demo/build_static.py              -> docs/index.html   (a complete document; serve docs/ with GitHub Pages or any static host)
  python demo/build_static.py --artifact PATH
                                           -> PATH              (the same page as a fragment for the claude.ai Artifact host, which supplies
                                                                 its own <html>/<head>/<body>; used to publish a demo URL)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _html_with_data() -> str:
    html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    data = json.dumps(json.loads((ROOT / "demo" / "recordings.json").read_text(encoding="utf-8")), separators=(",", ":"))
    data = data.replace("</", "<\\/")                     # keep the JSON from closing the script tag
    assert "/*__DATA__*/null" in html
    return html.replace("/*__DATA__*/null", data)


def build() -> Path:
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(_html_with_data(), encoding="utf-8")
    return out


def build_artifact(path: Path) -> Path:
    """Fragment for the Artifact host: no doctype/html/head/body tags, the host's viewport meta (the charset meta stays: harmless, and safe on a host that sends none), side gutters that clear a phone's system bars."""
    s = _html_with_data()
    s = re.sub(r"<!doctype html>\s*", "", s, flags=re.I)
    s = re.sub(r"</?(html|head|body)\b[^>]*>\s*", "", s)
    s = re.sub(r'<meta name="viewport"[^>]*>\s*', "", s)
    s = s.replace("<title>", '<meta name="claude:viewport-fit-sides" content="cover">\n<title>', 1)
    old = ".wrap{max-width:1180px;margin:0 auto;padding:20px 16px 48px}"
    new = (".wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-left:max(16px, env(safe-area-inset-left, 0px));"
           "padding-right:max(16px, env(safe-area-inset-right, 0px))}")
    assert old in s
    s = s.replace(old, new)
    # native controls and scrollbars follow the dark palette in both dark blocks
    s = s.replace("--exp:#6ea8ff; --got:#ff8a4c; --code:#1d222c; --add:#12301f; --del:#3a1714; --accent:#6ea8ff;",
                  "--exp:#6ea8ff; --got:#ff8a4c; --code:#1d222c; --add:#12301f; --del:#3a1714; --accent:#6ea8ff; color-scheme:dark;")
    path = Path(path)
    path.write_text(s, encoding="utf-8")
    return path


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--artifact":
        p = build_artifact(Path(sys.argv[2]))
    else:
        p = build()
    print("wrote", p, f"{p.stat().st_size // 1024} KB")
