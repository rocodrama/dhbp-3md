from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threemd.manifests import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a 3MD canonical JSONL manifest.")
    parser.add_argument("manifest")
    args = parser.parse_args()

    count = sum(1 for _ in load_jsonl(args.manifest))
    print(f"valid records: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
