from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.data import load_matches
from tennis_model.features import DEFAULT_ELO_CONFIG, EloConfig, build_training_frame_with_state
from tennis_model.utr import UTRTracker, load_utr_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rich live player state artifact from ATP history.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts_live")
    parser.add_argument("--utr-history-csv", default=None)
    parser.add_argument("--utr-alias-csv", default=None)
    parser.add_argument("--elo-config-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = load_matches(args.data_dir)
    utr_tracker = None
    if args.utr_history_csv:
        utr_tracker = UTRTracker.from_history(
            load_utr_history(args.utr_history_csv, alias_csv=args.utr_alias_csv)
        )

    elo_config = DEFAULT_ELO_CONFIG
    if args.elo_config_json:
        with Path(args.elo_config_json).open("r", encoding="utf-8") as handle:
            elo_config = EloConfig(**json.load(handle))

    _, snapshot, live_state, pair_history = build_training_frame_with_state(
        matches,
        utr_tracker=utr_tracker,
        elo_config=elo_config,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "player_snapshot.csv"
    live_state_path = output_dir / "player_live_state.csv"
    pair_history_path = output_dir / "pair_history.csv"
    snapshot.to_csv(snapshot_path, index=False)
    live_state.to_csv(live_state_path, index=False)
    pair_history.to_csv(pair_history_path, index=False)
    print(f"Wrote {snapshot_path}")
    print(f"Wrote {live_state_path}")
    print(f"Wrote {pair_history_path}")


if __name__ == "__main__":
    main()
