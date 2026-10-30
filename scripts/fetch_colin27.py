"""Download the Colin27 brain (ch2.nii.gz, 1.9 MB) used by the real-scan checks to _work/data/. Needs network; no tokens.

It is not redistributed in this repository. The file comes from the ANTsPy repository at a pinned commit and is checked against its SHA-256,
so every clone validates on the same four scans. Without it the real-scan checks run on the two nibabel scans only.

  python scripts/fetch_colin27.py"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract.realscan import COLIN  # noqa: E402

URL = "https://raw.githubusercontent.com/ANTsX/ANTsPy/1e73b82fd23a9c39d80918739991da41f290cd9d/data/ch2.nii.gz"
SHA256 = "52a989c2f85f67550dae601ae4eb9c79db9e92510e84a6f2f9dde56f150df8ce"


def main() -> int:
    if COLIN.exists() and hashlib.sha256(COLIN.read_bytes()).hexdigest() == SHA256:
        print(f"already present: {COLIN}")
        return 0
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch ({digest}); not saved", file=sys.stderr)
        return 1
    COLIN.parent.mkdir(parents=True, exist_ok=True)
    COLIN.write_bytes(data)
    print(f"saved {COLIN} ({len(data):,} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
