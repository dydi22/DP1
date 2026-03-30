from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.bracket import simulate_tournament
from tennis_model.draws import fetch_atp_draw
from tennis_model.names import normalize_player_name
from tennis_model.predict import load_pair_history, load_snapshot, predict_match_probability_with_model


DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts_plus_stats" / "model.joblib"
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "artifacts_plus_stats" / "player_snapshot.csv"
DEFAULT_PAIR_HISTORY_PATH = PROJECT_ROOT / "artifacts_plus_stats" / "pair_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch an ATP draw, write the bracket, score the first round, and simulate the tournament."
    )
    parser.add_argument("--draw-url", required=True)
    parser.add_argument("--match-type", default=None, help="Optional ATP matchtype query value such as singles or qualifiersingles.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--pair-history-path", default=str(DEFAULT_PAIR_HISTORY_PATH))
    parser.add_argument("--output-dir", default="artifacts/atp_workflow")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--tourney-level", default=None)
    parser.add_argument("--draw-size", type=int, default=None)
    parser.add_argument("--tournament-date", default=None)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Allow simulations even when the draw still contains unresolved placeholder names.",
    )
    return parser.parse_args()


def score_first_round(
    model,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    draw_pairs: list[tuple[str, str]],
    *,
    placeholder_names: set[str],
    match_date: str | None,
    surface: str,
    best_of: int,
    round_name: str,
    tourney_level: str,
    draw_size: int,
) -> pd.DataFrame:
    known_names = set(snapshot["_lookup_name"])
    rows: list[dict[str, object]] = []
    for player_1, player_2 in draw_pairs:
        lookup_1 = normalize_player_name(player_1)
        lookup_2 = normalize_player_name(player_2)
        player_1_in_snapshot = player_1 == "BYE" or lookup_1 in known_names
        player_2_in_snapshot = player_2 == "BYE" or lookup_2 in known_names
        model_ready = player_1 not in placeholder_names and player_2 not in placeholder_names

        probability = None
        favorite = None
        if player_1 == "BYE":
            probability = 0.0
            favorite = player_2
        elif player_2 == "BYE":
            probability = 1.0
            favorite = player_1
        elif model_ready:
            probability = predict_match_probability_with_model(
                model,
                snapshot,
                player_1=player_1,
                player_2=player_2,
                match_date=match_date,
                surface=surface,
                best_of=best_of,
                round_name=round_name,
                tourney_level=tourney_level,
                draw_size=draw_size,
                pair_history=pair_history,
            )
            favorite = player_1 if probability >= 0.5 else player_2
        else:
            probability = None
            favorite = None

        rows.append(
            {
                "player_1": player_1,
                "player_2": player_2,
                "player_1_win_probability": probability,
                "favorite": favorite,
                "model_ready": model_ready,
                "player_1_in_snapshot": player_1_in_snapshot,
                "player_2_in_snapshot": player_2_in_snapshot,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    draw = fetch_atp_draw(args.draw_url, match_type=args.match_type)
    tourney_level = args.tourney_level or draw.inferred_tourney_level or "A"
    best_of = args.best_of or draw.inferred_best_of or 3
    tournament_date = args.tournament_date or draw.tournament_date
    draw_size = args.draw_size or draw.draw_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bracket_path = output_dir / "bracket.csv"
    pd.DataFrame(draw.first_round_pairs, columns=["player_1", "player_2"]).to_csv(bracket_path, index=False)

    metadata = draw.to_metadata()
    metadata["surface"] = args.surface
    metadata["best_of"] = best_of
    metadata["tourney_level"] = tourney_level
    metadata["draw_size_used"] = draw_size
    metadata["tournament_date_used"] = tournament_date

    metadata_path = output_dir / "draw_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    snapshot = load_snapshot(args.snapshot_path)
    pair_history = load_pair_history(args.pair_history_path) if Path(args.pair_history_path).exists() else None
    import joblib

    model = joblib.load(args.model_path)

    placeholder_names = set(draw.placeholder_names)
    first_round_predictions = score_first_round(
        model,
        snapshot,
        pair_history,
        draw.first_round_pairs,
        placeholder_names=placeholder_names,
        match_date=tournament_date,
        surface=args.surface,
        best_of=best_of,
        round_name=draw.first_round_label,
        tourney_level=tourney_level,
        draw_size=draw_size,
    )
    first_round_path = output_dir / "first_round_predictions.csv"
    first_round_predictions.to_csv(first_round_path, index=False)

    unresolved_rows = first_round_predictions.loc[~first_round_predictions["model_ready"]]
    probabilities_path = output_dir / "tournament_probabilities.csv"
    if unresolved_rows.empty or args.allow_unresolved:
        probabilities = simulate_tournament(
            model_path=args.model_path,
            snapshot=snapshot,
            pair_history=pair_history,
            first_round_pairs=draw.first_round_pairs,
            tournament_date=tournament_date,
            surface=args.surface,
            best_of=best_of,
            tourney_level=tourney_level,
            draw_size=draw_size,
            simulations=args.simulations,
            round_labels=draw.round_labels,
            terminal_label=draw.terminal_label,
        )
        probabilities.to_csv(probabilities_path, index=False)
        print(f"Wrote {probabilities_path}")
    else:
        print(
            "Skipped tournament simulation because some draw slots are still unresolved placeholders. "
            "Use --allow-unresolved if you want to simulate anyway."
        )

    print(f"Wrote {bracket_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {first_round_path}")


if __name__ == "__main__":
    main()
