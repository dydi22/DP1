from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.atp_results import fetch_atp_completed_results, results_url_from_draw_url
from tennis_model.draws import fetch_atp_draw
from tennis_model.live import (
    apply_completed_matches,
    audit_completed_predictions,
    build_pending_matches,
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
        description=(
            "Fetch an ATP draw, scrape completed ATP results and official match stats, "
            "update live player state, and simulate the remaining tournament."
        )
    )
    parser.add_argument("--draw-url", required=True)
    parser.add_argument("--match-type", default=None, help="Optional ATP draw matchtype query, such as qualifiersingles.")
    parser.add_argument("--results-url", default=None, help="Optional override for the ATP results page URL.")
    parser.add_argument("--results-match-type", default="singles", help="ATP results-page matchType query value. Defaults to singles.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--live-state-path", default=str(DEFAULT_LIVE_STATE_PATH))
    parser.add_argument("--pair-history-path", default=str(DEFAULT_PAIR_HISTORY_PATH))
    parser.add_argument("--output-dir", default="artifacts_live/atp_auto_update")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--surface", required=True)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--tourney-level", default=None)
    parser.add_argument("--draw-size", type=int, default=None)
    parser.add_argument("--tournament-date", default=None)
    parser.add_argument("--simulations", type=int, default=5000)
    return parser.parse_args()


def _write_draw_artifacts(
    *,
    output_dir: Path,
    draw,
    surface: str,
    best_of: int,
    tourney_level: str,
    draw_size: int,
    tournament_date: str | None,
) -> tuple[Path, Path]:
    bracket_path = output_dir / "bracket.csv"
    pd.DataFrame(draw.first_round_pairs, columns=["player_1", "player_2"]).to_csv(bracket_path, index=False)

    metadata = draw.to_metadata()
    metadata["surface"] = surface
    metadata["best_of"] = best_of
    metadata["tourney_level"] = tourney_level
    metadata["draw_size_used"] = draw_size
    metadata["tournament_date_used"] = tournament_date

    metadata_path = output_dir / "draw_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return bracket_path, metadata_path


def main() -> None:
    args = parse_args()

    draw = fetch_atp_draw(args.draw_url, match_type=args.match_type)
    tourney_level = args.tourney_level or draw.inferred_tourney_level or "A"
    best_of = args.best_of or draw.inferred_best_of or 3
    draw_size = args.draw_size or draw.draw_size
    tournament_date = args.tournament_date or draw.tournament_date
    results_url = args.results_url or results_url_from_draw_url(args.draw_url)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bracket_path, metadata_path = _write_draw_artifacts(
        output_dir=output_dir,
        draw=draw,
        surface=args.surface,
        best_of=best_of,
        tourney_level=tourney_level,
        draw_size=draw_size,
        tournament_date=tournament_date,
    )

    allowed_rounds = set(draw.round_labels)
    completed_results = fetch_atp_completed_results(
        results_url,
        match_type=args.results_match_type,
        allowed_rounds=allowed_rounds,
        surface=args.surface,
        best_of=best_of,
        tourney_level=tourney_level,
    )
    completed_results_path = output_dir / "completed_results_auto.csv"
    completed_results.to_csv(completed_results_path, index=False)

    live_state = load_live_state(args.live_state_path)
    pair_history = load_pair_history_state(args.pair_history_path) if Path(args.pair_history_path).exists() else pd.DataFrame()
    updated_live_state, updated_snapshot, updated_pair_history = apply_completed_matches(
        live_state,
        pair_history,
        completed_results,
        default_surface=args.surface,
        default_best_of=best_of,
        default_tourney_level=tourney_level,
    )

    pending_matches, active_rounds, current_stage = build_pending_matches(
        first_round_pairs=draw.first_round_pairs,
        round_labels=draw.round_labels,
        completed_results=completed_results,
    )

    updated_live_state_path = output_dir / "updated_live_state.csv"
    updated_snapshot_path = output_dir / "updated_player_snapshot.csv"
    updated_pair_history_path = output_dir / "updated_pair_history.csv"
    completed_audit_path = output_dir / "completed_match_audit.csv"
    current_bracket_path = output_dir / "current_round_bracket.csv"
    current_predictions_path = output_dir / "current_round_predictions.csv"
    remaining_probabilities_path = output_dir / "remaining_tournament_probabilities.csv"
    summary_path = output_dir / "auto_update_summary.json"

    model = joblib.load(args.model_path)
    completed_match_audit = audit_completed_predictions(
        model,
        live_state,
        pair_history,
        completed_results,
        default_surface=args.surface,
        default_best_of=best_of,
        default_tourney_level=tourney_level,
        draw_size=len(draw.first_round_pairs) * 2,
        round_labels=draw.round_labels,
    )

    updated_live_state.to_csv(updated_live_state_path, index=False)
    updated_snapshot.to_csv(updated_snapshot_path, index=False)
    updated_pair_history.to_csv(updated_pair_history_path, index=False)
    completed_match_audit.to_csv(completed_audit_path, index=False)
    history_path, history_bucket_path, history_summary_path = upsert_prediction_history(
        args.history_dir,
        completed_match_audit,
        source_name="auto_update_tournament",
    )
    runtime_snapshot = updated_snapshot.copy()
    runtime_snapshot["_lookup_name"] = runtime_snapshot["player_name"].map(normalize_player_name)

    summary: dict[str, object] = {
        "draw_url": args.draw_url,
        "results_url": results_url,
        "match_type": args.match_type,
        "results_match_type": args.results_match_type,
        "surface": args.surface,
        "best_of": best_of,
        "tourney_level": tourney_level,
        "draw_size": draw_size,
        "tournament_date": tournament_date,
        "completed_matches_scraped": int(len(completed_results)),
        "completed_matches_with_stats_errors": int(completed_results["stats_fetch_error"].notna().sum())
        if "stats_fetch_error" in completed_results.columns
        else 0,
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
        "terminal_label": draw.terminal_label,
        "placeholder_slots": draw.placeholder_names,
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
            surface=args.surface,
            best_of=best_of,
            tourney_level=tourney_level,
            draw_size=len(draw.first_round_pairs) * 2,
        )
        current_predictions.to_csv(current_predictions_path, index=False)

        remaining_probabilities = simulate_remaining_tournament(
            model_path=args.model_path,
            snapshot=runtime_snapshot,
            pair_history=updated_pair_history,
            first_round_pairs=draw.first_round_pairs,
            completed_results=completed_results,
            round_labels=draw.round_labels,
            terminal_label=draw.terminal_label,
            tournament_date=tournament_date,
            surface=args.surface,
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

    print(f"Wrote {bracket_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {completed_results_path}")
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
