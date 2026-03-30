from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.data import load_matches
from tennis_model.features import DEFAULT_ELO_CONFIG, EloConfig, build_training_frame_with_state
from tennis_model.modeling import (
    FEATURE_COLUMNS,
    benchmark_models,
    calibration_bucket_frame,
    error_slice_frame,
    evaluate_predictions,
    fit_calibrated_model,
    high_confidence_miss_frame,
    model_insight_frame,
    predict_frame_probabilities,
    save_artifacts,
    time_split,
)
from tennis_model.utr import UTRTracker, load_utr_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark ATP singles match models.")
    parser.add_argument("--data-dir", required=True, help="Path to the Jeff Sackmann tennis_atp repo.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for saved outputs.")
    parser.add_argument(
        "--utr-history-csv",
        default=None,
        help="Optional CSV of historical UTR ratings with columns player_name,rating_date,utr_singles.",
    )
    parser.add_argument(
        "--utr-alias-csv",
        default=None,
        help="Optional name mapping CSV with columns source_name,canonical_name for UTR alignment.",
    )
    parser.add_argument(
        "--validation-start-year",
        type=int,
        default=None,
        help="First season to reserve for validation. Default is test_start_year - 1.",
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=None,
        help="First season to reserve for test data. Default is latest available season.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--elo-config-json",
        default=None,
        help="Optional JSON file of Elo tuning parameters to use when rebuilding rating features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = load_matches(args.data_dir)
    utr_tracker = None
    if args.utr_history_csv:
        utr_history = load_utr_history(
            args.utr_history_csv,
            alias_csv=args.utr_alias_csv,
        )
        utr_tracker = UTRTracker.from_history(utr_history)

    elo_config = DEFAULT_ELO_CONFIG
    if args.elo_config_json:
        with Path(args.elo_config_json).open("r", encoding="utf-8") as handle:
            elo_config = EloConfig(**json.load(handle))

    training_frame, snapshot, live_state, pair_history = build_training_frame_with_state(
        matches,
        utr_tracker=utr_tracker,
        elo_config=elo_config,
    )

    selected_model_name, _, benchmark_frame, _ = benchmark_models(
        training_frame,
        validation_start_year=args.validation_start_year,
        test_start_year=args.test_start_year,
        random_state=args.random_state,
    )
    train_df, validation_df, test_df = time_split(
        training_frame,
        validation_start_year=args.validation_start_year,
        test_start_year=args.test_start_year,
    )
    calibration_training_frame = pd.concat([train_df, validation_df], ignore_index=True)
    evaluation_pipeline = fit_calibrated_model(
        calibration_training_frame,
        model_name=selected_model_name,
        random_state=args.random_state,
    )
    calibrated_test_probabilities = predict_frame_probabilities(evaluation_pipeline, test_df)
    calibrated_test_metrics = evaluate_predictions(test_df["target"], calibrated_test_probabilities)
    predictions = test_df[
        ["match_date", "player_1", "player_2", "target", "surface", "round", "best_of", "tourney_level"]
    ].copy()
    predictions["predicted_probability"] = calibrated_test_probabilities.to_numpy()
    calibration_buckets = calibration_bucket_frame(test_df["target"], calibrated_test_probabilities)
    error_slices = error_slice_frame(predictions)
    high_confidence_misses = high_confidence_miss_frame(predictions)
    pipeline = fit_calibrated_model(
        training_frame,
        model_name=selected_model_name,
        random_state=args.random_state,
    )
    importances = model_insight_frame(
        pipeline,
        sample_features=training_frame[FEATURE_COLUMNS],
        sample_target=training_frame["target"],
        random_state=args.random_state,
    )

    save_artifacts(
        output_dir=args.output_dir,
        pipeline=pipeline,
        selected_model_name=selected_model_name,
        benchmark_frame=benchmark_frame,
        predictions=predictions,
        model_insights=importances,
        snapshot=snapshot,
        live_state=live_state,
        pair_history=pair_history,
        summary_metrics={
            **benchmark_frame.loc[benchmark_frame["model_name"] == selected_model_name].iloc[0].to_dict(),
            **{f"calibrated_test_{key}": value for key, value in calibrated_test_metrics.items()},
        },
        training_config={
            "data_dir": args.data_dir,
            "utr_history_csv": args.utr_history_csv,
            "utr_alias_csv": args.utr_alias_csv,
            "validation_start_year": args.validation_start_year,
            "test_start_year": args.test_start_year,
            "calibration_method": "sigmoid_cv_3",
            "random_state": args.random_state,
            "elo_config": elo_config.__dict__,
        },
        calibration_buckets=calibration_buckets,
        error_slices=error_slices,
        high_confidence_misses=high_confidence_misses,
    )

    print("Training complete.")
    print(f"Selected model: {selected_model_name}")
    best_row = benchmark_frame.loc[benchmark_frame["model_name"] == selected_model_name].iloc[0].to_dict()
    for key, value in best_row.items():
        if key == "model_name":
            continue
        print(f"{key}: {value:.4f}")
    for key, value in calibrated_test_metrics.items():
        print(f"calibrated_test_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
