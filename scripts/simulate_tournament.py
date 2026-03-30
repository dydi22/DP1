from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.bracket import load_bracket, simulate_tournament
from tennis_model.predict import load_pair_history, load_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a knockout tournament bracket.")
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--snapshot-path", default="artifacts/player_snapshot.csv")
    parser.add_argument("--pair-history-path", default="artifacts/pair_history.csv")
    parser.add_argument("--bracket-csv", required=True)
    parser.add_argument("--tournament-date", default=None, help="Optional YYYY-MM-DD tournament date.")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--best-of", required=True, type=int)
    parser.add_argument("--tourney-level", required=True)
    parser.add_argument("--draw-size", required=True, type=int)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument(
        "--round-labels",
        default=None,
        help="Optional comma-separated round labels such as Q1,Q2 or R128,R64,R32,R16,QF,SF,F.",
    )
    parser.add_argument("--terminal-label", default="Champion")
    parser.add_argument("--output-path", default="artifacts/bracket_probabilities.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot_path)
    pair_history = load_pair_history(args.pair_history_path) if Path(args.pair_history_path).exists() else None
    bracket = load_bracket(args.bracket_csv)

    probabilities = simulate_tournament(
        model_path=args.model_path,
        snapshot=snapshot,
        pair_history=pair_history,
        first_round_pairs=bracket,
        tournament_date=args.tournament_date,
        surface=args.surface,
        best_of=args.best_of,
        tourney_level=args.tourney_level,
        draw_size=args.draw_size,
        simulations=args.simulations,
        round_labels=[label.strip() for label in args.round_labels.split(",")] if args.round_labels else None,
        terminal_label=args.terminal_label,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
