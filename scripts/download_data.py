"""Fetch the development data used by the EDA report."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moex_portfolio.data import collect  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eda.json")
    args = parser.parse_args()
    collect(ROOT, json.loads(args.config.read_text()), offline=args.offline)


if __name__ == "__main__":
    main()
