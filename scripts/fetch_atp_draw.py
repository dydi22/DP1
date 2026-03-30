from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.draws import fetch_atp_draw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch an official ATP draw page and save it as a bracket CSV.")
    parser.add_argument("--draw-url", required=True)
    parser.add_argument("--match-type", default=None, help="Optional ATP matchtype query value such as singles or qualifiersingles.")
    parser.add_argument("--output-dir", default="artifacts/atp_draw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw = fetch_atp_draw(args.draw_url, match_type=args.match_type)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bracket_path = output_dir / "bracket.csv"
    pd.DataFrame(draw.first_round_pairs, columns=["player_1", "player_2"]).to_csv(bracket_path, index=False)

    metadata_path = output_dir / "draw_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(draw.to_metadata(), handle, indent=2)

    print(f"Wrote {bracket_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
