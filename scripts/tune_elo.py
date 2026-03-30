from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.data import load_matches
from tennis_model.features import DEFAULT_ELO_CONFIG, EloConfig, build_training_frame_with_state
from tennis_model.modeling import elo_expected_probability, evaluate_predictions, time_split


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None or value.strip() == "":
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe Elo parameter sweep using future-season validation.")
    parser.add_argument("--data-dir", required=True, help="Path to the ATP data directory.")
    parser.add_argument("--output-dir", default="artifacts_elo_tuning", help="Directory for Elo search outputs.")
    parser.add_argument("--validation-start-year", type=int, required=True, help="First validation year.")
    parser.add_argument("--test-start-year", type=int, required=True, help="First test year.")
    parser.add_argument("--base-k-values", default="24,28,32")
    parser.add_argument("--overall-half-life-values", default="180,240,300")
    parser.add_argument("--recent-half-life-values", default="90,120,150")
    parser.add_argument("--best-of-five-half-life-values", default="240,365")
    parser.add_argument("--inactivity-max-values", default="1.15,1.35")
    parser.add_argument("--recent-k-multiplier-values", default="1.15,1.25")
    parser.add_argument("--limit-combos", type=int, default=None, help="Optional cap on the number of grid combinations to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = load_matches(args.data_dir)

    base_k_values = parse_float_list(args.base_k_values, [24.0, 28.0, 32.0])
    overall_half_life_values = parse_float_list(args.overall_half_life_values, [180.0, 240.0, 300.0])
    recent_half_life_values = parse_float_list(args.recent_half_life_values, [90.0, 120.0, 150.0])
    best_of_five_half_life_values = parse_float_list(args.best_of_five_half_life_values, [240.0, 365.0])
    inactivity_max_values = parse_float_list(args.inactivity_max_values, [1.15, 1.35])
    recent_k_multiplier_values = parse_float_list(args.recent_k_multiplier_values, [1.15, 1.25])

    combinations = list(
        product(
            base_k_values,
            overall_half_life_values,
            recent_half_life_values,
            best_of_five_half_life_values,
            inactivity_max_values,
            recent_k_multiplier_values,
        )
    )
    if args.limit_combos is not None:
        combinations = combinations[: args.limit_combos]

    rows: list[dict[str, float | str | int]] = []
    best_row: dict[str, float | str | int] | None = None
    best_config: EloConfig | None = None

    for index, (base_k, overall_half_life, recent_half_life, best_of_five_half_life, inactivity_max, recent_k_multiplier) in enumerate(combinations, start=1):
        elo_config = EloConfig(
            base_k=base_k,
            overall_half_life_days=overall_half_life,
            recent_half_life_days=recent_half_life,
            best_of_five_half_life_days=best_of_five_half_life,
            surface_half_life_days=overall_half_life,
            inactivity_max_multiplier=inactivity_max,
            recent_k_multiplier=recent_k_multiplier,
        )
        training_frame, _, _, _ = build_training_frame_with_state(
            matches,
            elo_config=elo_config,
        )
        train_df, validation_df, test_df = time_split(
            training_frame,
            validation_start_year=args.validation_start_year,
            test_start_year=args.test_start_year,
        )
        _ = train_df
        validation_probabilities = elo_expected_probability(
            validation_df["best_of_context_elo_gap"].fillna(validation_df["elo_gap"]).fillna(0.0)
        )
        test_probabilities = elo_expected_probability(
            test_df["best_of_context_elo_gap"].fillna(test_df["elo_gap"]).fillna(0.0)
        )
        validation_metrics = evaluate_predictions(validation_df["target"], validation_probabilities)
        test_metrics = evaluate_predictions(test_df["target"], test_probabilities)
        row = {
            "combo_index": index,
            "base_k": base_k,
            "overall_half_life_days": overall_half_life,
            "recent_half_life_days": recent_half_life,
            "best_of_five_half_life_days": best_of_five_half_life,
            "inactivity_max_multiplier": inactivity_max,
            "recent_k_multiplier": recent_k_multiplier,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        rows.append(row)
        if best_row is None or (
            row["validation_log_loss"],
            row["validation_brier_score"],
            -row["validation_roc_auc"],
        ) < (
            best_row["validation_log_loss"],
            best_row["validation_brier_score"],
            -best_row["validation_roc_auc"],
        ):
            best_row = row
            best_config = elo_config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "validation_brier_score", "validation_roc_auc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    results_frame.to_csv(output_dir / "elo_search_results.csv", index=False)

    best_summary = {
        "data_dir": args.data_dir,
        "validation_start_year": args.validation_start_year,
        "test_start_year": args.test_start_year,
        "combination_count": len(combinations),
        "default_elo_config": DEFAULT_ELO_CONFIG.__dict__,
        "best_result": best_row,
        "best_elo_config": best_config.__dict__ if best_config is not None else None,
    }
    with (output_dir / "best_elo_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)

    if best_config is not None:
        with (output_dir / "elo_config_only.json").open("w", encoding="utf-8") as handle:
            json.dump(best_config.__dict__, handle, indent=2)

    print("Elo tuning complete.")
    print(json.dumps(best_summary, indent=2))


if __name__ == "__main__":
    main()
