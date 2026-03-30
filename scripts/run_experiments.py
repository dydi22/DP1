from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.data import load_matches
from tennis_model.features import build_training_frame_with_state
from tennis_model.modeling import (
    FEATURE_COLUMNS,
    RATING_FEATURE_COLUMNS,
    benchmark_models,
    calibration_bucket_frame,
    error_slice_frame,
    evaluate_predictions,
    fit_calibrated_model,
    high_confidence_miss_frame,
    predict_frame_probabilities,
    time_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe tennis model ablation experiments.")
    parser.add_argument("--data-dir", required=True, help="Path to the ATP match data directory.")
    parser.add_argument("--output-dir", default="artifacts_experiments", help="Directory for experiment outputs.")
    parser.add_argument("--validation-start-year", type=int, default=None)
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def feature_sets() -> dict[str, list[str]]:
    rating_core = list(RATING_FEATURE_COLUMNS)
    rating_plus_form = rating_core + [
        "recent_form_gap",
        "recent_adjusted_form_gap",
        "recent_minutes_total_gap",
        "recent_minutes_average_gap",
        "serve_win_rate_gap",
        "return_win_rate_gap",
        "surface_serve_win_rate_gap",
        "surface_return_win_rate_gap",
        "recent_serve_win_rate_gap",
        "recent_return_win_rate_gap",
    ]
    deduped_rating_plus_form = list(dict.fromkeys(rating_plus_form))
    matchup_enriched = deduped_rating_plus_form + [
        "player_1_is_lefty",
        "player_2_is_lefty",
        "lefty_mismatch",
        "serve_vs_return_edge_gap",
        "surface_serve_vs_return_edge_gap",
        "recent_serve_vs_return_edge_gap",
        "second_serve_pressure_gap",
        "ace_vs_return_pressure_gap",
        "break_pressure_gap",
        "surface_form_synergy_gap",
        "recent_surface_form_synergy_gap",
    ]
    return {
        "rating_core": rating_core,
        "rating_plus_form": deduped_rating_plus_form,
        "matchup_enriched": list(dict.fromkeys(matchup_enriched)),
        "full_feature_set": FEATURE_COLUMNS,
    }


def main() -> None:
    args = parse_args()
    matches = load_matches(args.data_dir)
    training_frame, _, _, _ = build_training_frame_with_state(matches)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for feature_set_name, feature_columns in feature_sets().items():
        selected_model_name, _, benchmark_frame, _ = benchmark_models(
            training_frame,
            validation_start_year=args.validation_start_year,
            test_start_year=args.test_start_year,
            random_state=args.random_state,
            feature_columns=feature_columns,
        )

        train_df, validation_df, test_df = time_split(
            training_frame,
            validation_start_year=args.validation_start_year,
            test_start_year=args.test_start_year,
        )
        calibration_training_frame = pd.concat([train_df, validation_df], ignore_index=True)
        calibrated_model = fit_calibrated_model(
            calibration_training_frame,
            model_name=selected_model_name,
            random_state=args.random_state,
            feature_columns=feature_columns,
        )
        calibrated_test_probabilities = predict_frame_probabilities(
            calibrated_model,
            test_df,
            feature_columns=feature_columns,
        )
        calibrated_metrics = evaluate_predictions(test_df["target"], calibrated_test_probabilities)

        benchmark_frame = benchmark_frame.copy()
        benchmark_frame.insert(0, "feature_set", feature_set_name)
        benchmark_frame.to_csv(output_dir / f"{feature_set_name}_benchmarks.csv", index=False)
        experiment_rows.extend(benchmark_frame.to_dict("records"))

        prediction_frame = test_df[
            ["match_date", "player_1", "player_2", "target", "surface", "round", "best_of", "tourney_level"]
        ].copy()
        prediction_frame["predicted_probability"] = calibrated_test_probabilities.to_numpy()
        prediction_frame.to_csv(output_dir / f"{feature_set_name}_test_predictions.csv", index=False)

        calibration_bucket_frame(
            test_df["target"],
            calibrated_test_probabilities,
        ).to_csv(output_dir / f"{feature_set_name}_calibration_buckets.csv", index=False)
        error_slice_frame(prediction_frame).to_csv(output_dir / f"{feature_set_name}_error_slices.csv", index=False)
        high_confidence_miss_frame(prediction_frame).to_csv(
            output_dir / f"{feature_set_name}_high_confidence_misses.csv",
            index=False,
        )

        selected_rows.append(
            {
                "feature_set": feature_set_name,
                "selected_model": selected_model_name,
                **calibrated_metrics,
                "feature_count": len(feature_columns),
            }
        )

    pd.DataFrame(experiment_rows).to_csv(output_dir / "all_experiment_benchmarks.csv", index=False)
    selected_frame = pd.DataFrame(selected_rows).sort_values(
        ["log_loss", "brier_score", "roc_auc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    selected_frame.to_csv(output_dir / "experiment_summary.csv", index=False)

    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_dir": args.data_dir,
                "validation_start_year": args.validation_start_year,
                "test_start_year": args.test_start_year,
                "random_state": args.random_state,
                "feature_sets": {name: columns for name, columns in feature_sets().items()},
            },
            handle,
            indent=2,
        )

    print("Experiment run complete.")
    print(selected_frame.to_string(index=False))


if __name__ == "__main__":
    main()
