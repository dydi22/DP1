from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.data import load_matches
from tennis_model.features import build_training_frame_with_state
from tennis_model.modeling import (
    FEATURE_COLUMNS,
    evaluate_predictions,
    logistic_preprocessor,
    one_hot_preprocessor,
    ordinal_preprocessor,
    split_feature_columns,
    time_split,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency/runtime
    XGBClassifier = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe hyperparameter tuning with a strict time-based validation split.")
    parser.add_argument("--data-dir", required=True, help="Path to the ATP data directory.")
    parser.add_argument("--output-dir", default="artifacts_tuning", help="Directory to save tuning results.")
    parser.add_argument("--validation-start-year", type=int, required=True, help="First validation year.")
    parser.add_argument("--test-start-year", type=int, required=True, help="First test year.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def hist_gradient_candidates(random_state: int) -> list[tuple[str, dict[str, Any], Any]]:
    grids = product(
        [0.03, 0.05],
        [4, 6, 8],
        [200, 300],
        [30, 50, 80],
        [0.0, 0.1],
    )
    rows: list[tuple[str, dict[str, Any], Any]] = []
    for learning_rate, max_depth, max_iter, min_samples_leaf, l2_regularization in grids:
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "max_iter": max_iter,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "random_state": random_state,
        }
        rows.append(("hist_gradient_boosting", params, HistGradientBoostingClassifier(**params)))
    return rows


def random_forest_candidates(random_state: int) -> list[tuple[str, dict[str, Any], Any]]:
    grids = product(
        [250, 400],
        [10, 25, 50],
        [None, 14],
        ["sqrt", 0.5],
    )
    rows: list[tuple[str, dict[str, Any], Any]] = []
    for n_estimators, min_samples_leaf, max_depth, max_features in grids:
        params = {
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            "max_features": max_features,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": random_state,
        }
        rows.append(("random_forest", params, RandomForestClassifier(**params)))
    return rows


def xgboost_candidates(random_state: int) -> list[tuple[str, dict[str, Any], Any]]:
    if XGBClassifier is None:
        return []
    grids = product(
        [0.03, 0.05],
        [4, 5, 6],
        [250, 350],
        [0.8, 0.9],
        [0.8, 0.9],
    )
    rows: list[tuple[str, dict[str, Any], Any]] = []
    for learning_rate, max_depth, n_estimators, subsample, colsample_bytree in grids:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": 3,
            "reg_lambda": 1.2,
            "random_state": random_state,
            "n_jobs": 1,
        }
        rows.append(("xgboost", params, XGBClassifier(**params)))
    return rows


def build_pipeline(model_name: str, estimator: Any) -> Pipeline:
    numeric_features, categorical_features = split_feature_columns(FEATURE_COLUMNS)
    if model_name == "hist_gradient_boosting" or model_name == "xgboost":
        preprocessor = ordinal_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
    else:
        preprocessor = one_hot_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def main() -> None:
    args = parse_args()
    matches = load_matches(args.data_dir)
    training_frame, _, _, _ = build_training_frame_with_state(matches)
    train_df, validation_df, test_df = time_split(
        training_frame,
        validation_start_year=args.validation_start_year,
        test_start_year=args.test_start_year,
    )

    candidates = (
        hist_gradient_candidates(args.random_state)
        + random_forest_candidates(args.random_state)
        + xgboost_candidates(args.random_state)
    )

    rows: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None

    for model_name, params, estimator in candidates:
        try:
            pipeline = build_pipeline(model_name, estimator)
            pipeline.fit(train_df[FEATURE_COLUMNS], train_df["target"])
            validation_probabilities = pd.Series(
                pipeline.predict_proba(validation_df[FEATURE_COLUMNS])[:, 1],
                index=validation_df.index,
            )
            test_probabilities = pd.Series(
                pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1],
                index=test_df.index,
            )
            validation_metrics = evaluate_predictions(validation_df["target"], validation_probabilities)
            test_metrics = evaluate_predictions(test_df["target"], test_probabilities)
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(exc),
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
            continue

        row = {
            "model_name": model_name,
            "status": "ok",
            "params_json": json.dumps(params, sort_keys=True),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        rows.append(row)
        if best_entry is None or (
            row["validation_log_loss"],
            row["validation_brier_score"],
            -row["validation_roc_auc"],
        ) < (
            best_entry["validation_log_loss"],
            best_entry["validation_brier_score"],
            -best_entry["validation_roc_auc"],
        ):
            best_entry = row
            best_pipeline = clone(pipeline)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(rows).sort_values(
        ["status", "validation_log_loss", "validation_brier_score", "validation_roc_auc"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    results_frame.to_csv(output_dir / "hyperparameter_search_results.csv", index=False)

    summary = {
        "data_dir": args.data_dir,
        "validation_start_year": args.validation_start_year,
        "test_start_year": args.test_start_year,
        "random_state": args.random_state,
        "search_models": sorted(set(results_frame["model_name"].tolist())),
    }
    if best_entry is not None:
        summary["best_model_name"] = best_entry["model_name"]
        summary["best_params"] = json.loads(best_entry["params_json"])
        summary["best_validation_log_loss"] = best_entry["validation_log_loss"]
        summary["best_validation_brier_score"] = best_entry["validation_brier_score"]
        summary["best_validation_roc_auc"] = best_entry["validation_roc_auc"]
        summary["best_test_log_loss"] = best_entry["test_log_loss"]
        summary["best_test_roc_auc"] = best_entry["test_roc_auc"]
    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Hyperparameter search complete.")
    if best_entry is None:
        print("No successful model fits.")
        return
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
