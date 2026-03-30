from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.predict import load_pair_history, load_snapshot, predict_match_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict P(player_1 wins) for a match.")
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--snapshot-path", default="artifacts/player_snapshot.csv")
    parser.add_argument("--pair-history-path", default="artifacts/pair_history.csv")
    parser.add_argument("--player-1", required=True)
    parser.add_argument("--player-2", required=True)
    parser.add_argument("--match-date", default=None, help="Optional YYYY-MM-DD match date.")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--best-of", required=True, type=int)
    parser.add_argument("--round", dest="round_name", required=True)
    parser.add_argument("--tourney-level", required=True)
    parser.add_argument("--draw-size", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot_path)
    pair_history = load_pair_history(args.pair_history_path) if Path(args.pair_history_path).exists() else None
    probability = predict_match_probability(
        args.model_path,
        snapshot,
        player_1=args.player_1,
        player_2=args.player_2,
        match_date=args.match_date,
        surface=args.surface,
        best_of=args.best_of,
        round_name=args.round_name,
        tourney_level=args.tourney_level,
        draw_size=args.draw_size,
        pair_history=pair_history,
    )

    print(
        json.dumps(
            {
                "player_1": args.player_1,
                "player_2": args.player_2,
                "match_date": args.match_date,
                "surface": args.surface,
                "best_of": args.best_of,
                "round": args.round_name,
                "tourney_level": args.tourney_level,
                "draw_size": args.draw_size,
                "player_1_win_probability": round(probability, 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
