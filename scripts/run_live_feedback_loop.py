from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.live import (
    apply_completed_matches,
    audit_completed_predictions,
    build_pending_matches,
    load_completed_results,
    load_draw_metadata,
    load_live_state,
    load_pair_history_state,
    score_current_round,
    simulate_remaining_tournament,
)
from tennis_model.names import normalize_player_name
from tennis_model.tracking import upsert_prediction_history


DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts_plus_stats" / "model.joblib"
DEFAULT_LIVE_STATE_PATH = PROJECT_ROOT / "artifacts_plus_stats" / "player_live_state.csv"
DEFAULT_PAIR_HISTORY_PATH = PROJECT_ROOT / "artifacts_plus_stats" / "pair_history.csv"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "artifacts_plus_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply completed match results to the live player state and re-simulate the remaining tournament."
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--live-state-path", default=str(DEFAULT_LIVE_STATE_PATH))
    parser.add_argument("--pair-history-path", default=str(DEFAULT_PAIR_HISTORY_PATH))
    parser.add_argument("--draw-metadata-json", required=True)
    parser.add_argument("--completed-results-csv", required=True)
    parser.add_argument("--output-dir", default="artifacts_live/feedback_loop")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--surface", default=None)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--tourney-level", default=None)
    parser.add_argument("--tournament-date", default=None)
    parser.add_argument("--simulations", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    live_state = load_live_state(args.live_state_path)
    pair_history = load_pair_history_state(args.pair_history_path) if Path(args.pair_history_path).exists() else pd.DataFrame()
    completed_results = load_completed_results(args.completed_results_csv)
    draw_metadata = load_draw_metadata(args.draw_metadata_json)

    surface = args.surface or draw_metadata.get("surface") or "Hard"
    best_of = args.best_of or int(draw_metadata.get("best_of") or draw_metadata.get("inferred_best_of") or 3)
    tourney_level = args.tourney_level or draw_metadata.get("tourney_level") or draw_metadata.get("inferred_tourney_level") or "A"
    tournament_date = args.tournament_date or draw_metadata.get("tournament_date_used") or draw_metadata.get("tournament_date")

    updated_live_state, updated_snapshot, updated_pair_history = apply_completed_matches(
        live_state,
        pair_history,
        completed_results,
        default_surface=surface,
        default_best_of=best_of,
        default_tourney_level=tourney_level,
    )

    first_round_pairs = [tuple(pair) for pair in draw_metadata["first_round_pairs"]]
    round_labels = list(draw_metadata["round_labels"])
    terminal_label = str(draw_metadata.get("terminal_label") or "Champion")

    pending_matches, active_rounds, current_stage = build_pending_matches(
        first_round_pairs=first_round_pairs,
        round_labels=round_labels,
        completed_results=completed_results,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updated_live_state_path = output_dir / "updated_live_state.csv"
    updated_snapshot_path = output_dir / "updated_player_snapshot.csv"
    updated_pair_history_path = output_dir / "updated_pair_history.csv"
    completed_audit_path = output_dir / "completed_match_audit.csv"
    current_bracket_path = output_dir / "current_round_bracket.csv"
    current_predictions_path = output_dir / "current_round_predictions.csv"
    remaining_probabilities_path = output_dir / "remaining_tournament_probabilities.csv"
    summary_path = output_dir / "feedback_summary.json"

    model = joblib.load(args.model_path)
    completed_match_audit = audit_completed_predictions(
        model,
        live_state,
        pair_history,
        completed_results,
        default_surface=surface,
        default_best_of=best_of,
        default_tourney_level=tourney_level,
        draw_size=len(first_round_pairs) * 2,
        round_labels=round_labels,
    )

    updated_live_state.to_csv(updated_live_state_path, index=False)
    updated_snapshot.to_csv(updated_snapshot_path, index=False)
    updated_pair_history.to_csv(updated_pair_history_path, index=False)
    completed_match_audit.to_csv(completed_audit_path, index=False)
    history_path, history_bucket_path, history_summary_path = upsert_prediction_history(
        args.history_dir,
        completed_match_audit,
        source_name="run_live_feedback_loop",
    )
    runtime_snapshot = updated_snapshot.copy()
    runtime_snapshot["_lookup_name"] = runtime_snapshot["player_name"].map(normalize_player_name)

    summary: dict[str, object] = {
        "surface": surface,
        "best_of": best_of,
        "tourney_level": tourney_level,
        "tournament_date": tournament_date,
        "completed_matches_applied": int(len(completed_results)),
        "completed_matches_scored": int(len(completed_match_audit)),
        "completed_match_accuracy": float(completed_match_audit["correct_pick"].mean())
        if not completed_match_audit.empty
        else None,
        "completed_match_avg_log_loss": float(completed_match_audit["log_loss"].mean())
        if not completed_match_audit.empty
        else None,
        "completed_match_avg_brier_error": float(completed_match_audit["brier_error"].mean())
        if not completed_match_audit.empty
        else None,
        "current_stage": current_stage,
        "active_rounds": active_rounds,
        "terminal_label": terminal_label,
        "prediction_history_path": str(history_path),
    }

    if not pending_matches.empty:
        pending_matches.to_csv(current_bracket_path, index=False)
        current_predictions = score_current_round(
            model,
            runtime_snapshot,
            updated_pair_history,
            pending_matches,
            match_date=tournament_date,
            surface=surface,
            best_of=best_of,
            tourney_level=tourney_level,
            draw_size=len(first_round_pairs) * 2,
        )
        current_predictions.to_csv(current_predictions_path, index=False)

        remaining_probabilities = simulate_remaining_tournament(
            model_path=args.model_path,
            snapshot=runtime_snapshot,
            pair_history=updated_pair_history,
            first_round_pairs=first_round_pairs,
            completed_results=completed_results,
            round_labels=round_labels,
            terminal_label=terminal_label,
            tournament_date=tournament_date,
            surface=surface,
            best_of=best_of,
            tourney_level=tourney_level,
            simulations=args.simulations,
        )
        remaining_probabilities.to_csv(remaining_probabilities_path, index=False)
        summary["current_pair_count"] = int(len(pending_matches))
    else:
        summary["current_pair_count"] = 0
        summary["tournament_complete"] = True

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {updated_live_state_path}")
    print(f"Wrote {updated_snapshot_path}")
    print(f"Wrote {updated_pair_history_path}")
    print(f"Wrote {completed_audit_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {history_bucket_path}")
    print(f"Wrote {history_summary_path}")
    if not pending_matches.empty:
        print(f"Wrote {current_bracket_path}")
        print(f"Wrote {current_predictions_path}")
        print(f"Wrote {remaining_probabilities_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
