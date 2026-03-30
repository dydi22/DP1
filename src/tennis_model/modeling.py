from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency/runtime
    XGBClassifier = None


NUMERIC_FEATURES = [
    "best_of",
    "draw_size",
    "player_1_is_lefty",
    "player_2_is_lefty",
    "lefty_mismatch",
    "h2h_matches",
    "h2h_win_rate_edge",
    "surface_h2h_matches",
    "surface_h2h_win_rate_edge",
    "recent_h2h_matches",
    "recent_h2h_win_rate_edge",
    "days_since_h2h",
    "rank_gap",
    "rank_points_gap",
    "age_gap",
    "height_gap",
    "matches_gap",
    "wins_gap",
    "overall_win_rate_gap",
    "surface_matches_gap",
    "surface_wins_gap",
    "surface_win_rate_gap",
    "recent_form_gap",
    "recent_adjusted_form_gap",
    "elo_gap",
    "recent_elo_gap",
    "surface_elo_gap",
    "best_of_five_elo_gap",
    "best_of_context_elo_gap",
    "best_of_five_matches_gap",
    "best_of_five_win_rate_gap",
    "days_since_last_match_gap",
    "matches_last_30_days_gap",
    "recent_minutes_total_gap",
    "recent_minutes_average_gap",
    "serve_win_rate_gap",
    "return_win_rate_gap",
    "first_serve_in_rate_gap",
    "first_serve_win_rate_gap",
    "second_serve_win_rate_gap",
    "ace_rate_gap",
    "double_fault_rate_gap",
    "bp_save_rate_gap",
    "surface_serve_win_rate_gap",
    "surface_return_win_rate_gap",
    "recent_serve_win_rate_gap",
    "recent_return_win_rate_gap",
    "serve_vs_return_edge_gap",
    "surface_serve_vs_return_edge_gap",
    "recent_serve_vs_return_edge_gap",
    "second_serve_pressure_gap",
    "ace_vs_return_pressure_gap",
    "break_pressure_gap",
    "surface_form_synergy_gap",
    "recent_surface_form_synergy_gap",
    "utr_gap",
]

CATEGORICAL_FEATURES = ["surface", "indoor", "round", "tourney_level", "handedness_matchup"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RATING_NUMERIC_FEATURES = [
    "best_of",
    "draw_size",
    "h2h_matches",
    "h2h_win_rate_edge",
    "surface_h2h_matches",
    "surface_h2h_win_rate_edge",
    "recent_h2h_matches",
    "recent_h2h_win_rate_edge",
    "days_since_h2h",
    "rank_gap",
    "rank_points_gap",
    "age_gap",
    "elo_gap",
    "recent_elo_gap",
    "surface_elo_gap",
    "best_of_five_elo_gap",
    "best_of_context_elo_gap",
    "best_of_five_matches_gap",
    "best_of_five_win_rate_gap",
    "days_since_last_match_gap",
    "matches_last_30_days_gap",
]
RATING_CATEGORICAL_FEATURES = ["surface", "indoor", "round", "tourney_level", "handedness_matchup"]
RATING_FEATURE_COLUMNS = RATING_NUMERIC_FEATURES + RATING_CATEGORICAL_FEATURES


class ProbabilityBlendClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        weights: list[float] | None = None,
        *,
        include_elo: bool = True,
        elo_weight: float = 1.0,
        elo_feature: str = "best_of_context_elo_gap",
        fallback_elo_feature: str = "elo_gap",
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.include_elo = include_elo
        self.elo_weight = elo_weight
        self.elo_feature = elo_feature
        self.fallback_elo_feature = fallback_elo_feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilityBlendClassifier":
        self.estimators_ = [(name, clone(estimator).fit(X, y)) for name, estimator in self.estimators]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability_columns: list[np.ndarray] = []
        blend_weights: list[float] = []
        estimator_weights = self.weights or [1.0] * len(self.estimators_)
        for (name, estimator), weight in zip(self.estimators_, estimator_weights):
            _ = name
            probability_columns.append(estimator.predict_proba(X)[:, 1])
            blend_weights.append(float(weight))
        if self.include_elo:
            elo_gap = (
                X[self.elo_feature]
                .fillna(X[self.fallback_elo_feature])
                .fillna(0.0)
                .astype(float)
            )
            probability_columns.append(elo_expected_probability(elo_gap).to_numpy())
            blend_weights.append(float(self.elo_weight))
        stacked = np.vstack(probability_columns)
        blended = np.average(stacked, axis=0, weights=np.asarray(blend_weights, dtype=float))
        return np.column_stack([1.0 - blended, blended])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def split_feature_columns(
    feature_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    resolved_features = feature_columns or FEATURE_COLUMNS
    numeric_features = [column for column in resolved_features if column in NUMERIC_FEATURES]
    categorical_features = [column for column in resolved_features if column in CATEGORICAL_FEATURES]
    return numeric_features, categorical_features


def one_hot_preprocessor(
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    resolved_numeric = numeric_features or NUMERIC_FEATURES
    resolved_categorical = categorical_features or CATEGORICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                resolved_numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                resolved_categorical,
            ),
        ]
    )


def logistic_preprocessor(
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    resolved_numeric = numeric_features or NUMERIC_FEATURES
    resolved_categorical = categorical_features or CATEGORICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                resolved_numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                resolved_categorical,
            ),
        ]
    )


def ordinal_preprocessor(
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    resolved_numeric = numeric_features or NUMERIC_FEATURES
    resolved_categorical = categorical_features or CATEGORICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                resolved_numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                resolved_categorical,
            ),
        ]
    )


def build_candidate_pipelines(
    random_state: int,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    numeric_features, categorical_features = split_feature_columns(feature_columns)
    rating_numeric_features = [column for column in RATING_NUMERIC_FEATURES if column in numeric_features]
    rating_categorical_features = [column for column in RATING_CATEGORICAL_FEATURES if column in categorical_features]

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                logistic_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    solver="liblinear",
                    C=0.5,
                    random_state=random_state,
                ),
            ),
        ]
    )
    hist_gradient_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ordinal_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=250,
                    min_samples_leaf=50,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    rating_logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                logistic_preprocessor(
                    numeric_features=rating_numeric_features,
                    categorical_features=rating_categorical_features,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    solver="liblinear",
                    C=0.5,
                    random_state=random_state,
                ),
            ),
        ]
    )
    candidate_pipelines: dict[str, Any] = {
        "decision_tree": Pipeline(
            steps=[
                (
                    "preprocessor",
                    one_hot_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    ),
                ),
                (
                    "model",
                    DecisionTreeClassifier(
                        criterion="entropy",
                        max_depth=8,
                        min_samples_leaf=200,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logistic_regression": logistic_pipeline,
        "rating_logistic_baseline": rating_logistic_pipeline,
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    one_hot_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": hist_gradient_pipeline,
        "soft_voting_ensemble": VotingClassifier(
            estimators=[
                (
                    "logistic_regression",
                    Pipeline(
                        steps=[
                            (
                                "preprocessor",
                                logistic_preprocessor(
                                    numeric_features=numeric_features,
                                    categorical_features=categorical_features,
                                ),
                            ),
                            (
                                "model",
                                LogisticRegression(
                                    max_iter=1200,
                                    class_weight="balanced",
                                    solver="liblinear",
                                    C=0.5,
                                    random_state=random_state,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "hist_gradient_boosting",
                    Pipeline(
                        steps=[
                            (
                                "preprocessor",
                                ordinal_preprocessor(
                                    numeric_features=numeric_features,
                                    categorical_features=categorical_features,
                                ),
                            ),
                            (
                                "model",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.05,
                                    max_depth=6,
                                    max_iter=250,
                                    min_samples_leaf=50,
                                    l2_regularization=0.1,
                                    random_state=random_state,
                                ),
                            ),
                        ]
                    ),
                ),
            ],
            voting="soft",
            weights=[1.0, 2.0],
        ),
        "elo_blended_ensemble": ProbabilityBlendClassifier(
            estimators=[
                ("rating_logistic_baseline", clone(rating_logistic_pipeline)),
                ("hist_gradient_boosting", clone(hist_gradient_pipeline)),
            ],
            weights=[1.0, 2.0],
            include_elo=True,
            elo_weight=1.0,
        ),
    }
    if XGBClassifier is not None:
        xgboost_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ordinal_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        learning_rate=0.04,
                        max_depth=5,
                        n_estimators=350,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        reg_lambda=1.2,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        candidate_pipelines["xgboost"] = xgboost_pipeline
        candidate_pipelines["elo_blended_ensemble"] = ProbabilityBlendClassifier(
            estimators=[
                ("rating_logistic_baseline", clone(rating_logistic_pipeline)),
                ("hist_gradient_boosting", clone(hist_gradient_pipeline)),
                ("xgboost", clone(xgboost_pipeline)),
            ],
            weights=[1.0, 2.0, 2.0],
            include_elo=True,
            elo_weight=1.0,
        )
    return candidate_pipelines


def time_split(
    data: pd.DataFrame,
    validation_start_year: int | None = None,
    test_start_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = data["match_date"].dt.year
    latest_year = int(years.max())
    resolved_test_start_year = test_start_year if test_start_year is not None else latest_year
    resolved_validation_start_year = (
        validation_start_year
        if validation_start_year is not None
        else resolved_test_start_year - 1
    )

    train = data.loc[years < resolved_validation_start_year].copy()
    validation = data.loc[
        (years >= resolved_validation_start_year) & (years < resolved_test_start_year)
    ].copy()
    test = data.loc[years >= resolved_test_start_year].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Time split failed. Choose validation/test cutoffs that leave train, validation, and test data."
        )
    return train, validation, test


def evaluate_predictions(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    predicted_labels = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def elo_expected_probability(elo_gap: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + 10 ** (-(elo_gap.astype(float)) / 400.0))


def calibration_bucket_frame(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    bucket_count: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target": y_true.astype(int),
            "predicted_probability": probabilities.astype(float),
        }
    )
    frame["bucket"] = pd.cut(
        frame["predicted_probability"],
        bins=[index / bucket_count for index in range(bucket_count + 1)],
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for bucket, bucket_frame in frame.groupby("bucket", observed=False):
        if bucket_frame.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "matches": int(len(bucket_frame)),
                "avg_predicted_probability": float(bucket_frame["predicted_probability"].mean()),
                "actual_win_rate": float(bucket_frame["target"].mean()),
                "calibration_gap": float(bucket_frame["predicted_probability"].mean() - bucket_frame["target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def error_slice_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["predicted_label"] = (frame["predicted_probability"] >= 0.5).astype(int)
    frame["correct_pick"] = frame["predicted_label"] == frame["target"].astype(int)
    frame["confidence_bucket"] = pd.cut(
        (frame["predicted_probability"] - 0.5).abs(),
        bins=[0.0, 0.05, 0.10, 0.20, 0.50],
        labels=["coin_flip", "lean", "strong", "very_strong"],
        include_lowest=True,
    )

    slice_specs = {
        "surface": ["surface"],
        "round": ["round"],
        "best_of": ["best_of"],
        "confidence_bucket": ["confidence_bucket"],
        "surface_round": ["surface", "round"],
    }

    rows: list[dict[str, Any]] = []
    for slice_name, columns in slice_specs.items():
        available_columns = [column for column in columns if column in frame.columns]
        if len(available_columns) != len(columns):
            continue
        for group_values, group_frame in frame.groupby(available_columns, dropna=False, observed=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_dict = {column: value for column, value in zip(available_columns, group_values)}
            rows.append(
                {
                    "slice_name": slice_name,
                    **group_dict,
                    "matches": int(len(group_frame)),
                    "accuracy": float(group_frame["correct_pick"].mean()),
                    "avg_predicted_probability": float(group_frame["predicted_probability"].mean()),
                    "actual_win_rate": float(group_frame["target"].mean()),
                    "avg_brier_error": float(((group_frame["predicted_probability"] - group_frame["target"]) ** 2).mean()),
                }
            )
    return pd.DataFrame(rows)


def high_confidence_miss_frame(
    predictions: pd.DataFrame,
    *,
    probability_threshold: float = 0.65,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["predicted_label"] = (frame["predicted_probability"] >= 0.5).astype(int)
    frame["favorite"] = np.where(frame["predicted_probability"] >= 0.5, frame["player_1"], frame["player_2"])
    frame["favorite_win_probability"] = np.where(
        frame["predicted_probability"] >= 0.5,
        frame["predicted_probability"],
        1.0 - frame["predicted_probability"],
    )
    frame["actual_winner"] = np.where(frame["target"].astype(int) == 1, frame["player_1"], frame["player_2"])
    frame["correct_pick"] = frame["predicted_label"] == frame["target"].astype(int)
    frame["confidence"] = (frame["favorite_win_probability"] - 0.5) * 200.0
    misses = frame.loc[
        (~frame["correct_pick"]) & (frame["favorite_win_probability"] >= probability_threshold)
    ].copy()
    ordered_columns = [
        "match_date",
        "round",
        "surface",
        "player_1",
        "player_2",
        "actual_winner",
        "favorite",
        "favorite_win_probability",
        "confidence",
        "predicted_probability",
        "target",
        "best_of",
        "tourney_level",
    ]
    available_columns = [column for column in ordered_columns if column in misses.columns]
    return misses.sort_values(
        ["favorite_win_probability", "match_date"],
        ascending=[False, False],
    ).reset_index(drop=True)[available_columns]


def benchmark_models(
    training_frame: pd.DataFrame,
    *,
    validation_start_year: int | None,
    test_start_year: int | None,
    random_state: int,
    feature_columns: list[str] | None = None,
) -> tuple[str, Any, pd.DataFrame, pd.DataFrame]:
    train_df, validation_df, test_df = time_split(
        training_frame,
        validation_start_year=validation_start_year,
        test_start_year=test_start_year,
    )

    resolved_feature_columns = feature_columns or FEATURE_COLUMNS
    pipelines = build_candidate_pipelines(random_state, feature_columns=resolved_feature_columns)
    rows: list[dict[str, Any]] = []
    predictions_by_model: list[pd.DataFrame] = []

    elo_validation_probabilities = elo_expected_probability(validation_df["best_of_context_elo_gap"].fillna(validation_df["elo_gap"]).fillna(0.0))
    elo_test_probabilities = elo_expected_probability(test_df["best_of_context_elo_gap"].fillna(test_df["elo_gap"]).fillna(0.0))
    elo_validation_metrics = evaluate_predictions(validation_df["target"], elo_validation_probabilities)
    elo_test_metrics = evaluate_predictions(test_df["target"], elo_test_probabilities)
    rows.append(
        {
            "model_name": "elo_probability",
            **{f"validation_{key}": value for key, value in elo_validation_metrics.items()},
            **{f"test_{key}": value for key, value in elo_test_metrics.items()},
        }
    )
    elo_prediction_frame = test_df[["match_date", "player_1", "player_2", "target"]].copy()
    elo_prediction_frame["model_name"] = "elo_probability"
    elo_prediction_frame["predicted_probability"] = elo_test_probabilities.to_numpy()
    predictions_by_model.append(elo_prediction_frame)

    for model_name, pipeline in pipelines.items():
        pipeline.fit(train_df[resolved_feature_columns], train_df["target"])

        validation_probabilities = pd.Series(
            pipeline.predict_proba(validation_df[resolved_feature_columns])[:, 1]
        )
        test_probabilities = pd.Series(
            pipeline.predict_proba(test_df[resolved_feature_columns])[:, 1]
        )

        validation_metrics = evaluate_predictions(validation_df["target"], validation_probabilities)
        test_metrics = evaluate_predictions(test_df["target"], test_probabilities)
        rows.append(
            {
                "model_name": model_name,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

        prediction_frame = test_df[["match_date", "player_1", "player_2", "target"]].copy()
        prediction_frame["model_name"] = model_name
        prediction_frame["predicted_probability"] = test_probabilities.to_numpy()
        predictions_by_model.append(prediction_frame)

    benchmark_frame = pd.DataFrame(rows).sort_values(
        by=["validation_log_loss", "validation_brier_score", "validation_roc_auc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    selectable_frame = benchmark_frame.loc[benchmark_frame["model_name"] != "elo_probability"].reset_index(drop=True)
    best_model_name = str(selectable_frame.iloc[0]["model_name"])
    best_pipeline = None if best_model_name == "elo_probability" else pipelines[best_model_name]

    if best_pipeline is not None:
        full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
        best_pipeline.fit(full_train_df[resolved_feature_columns], full_train_df["target"])

    best_test_predictions = next(
        frame for frame in predictions_by_model if frame["model_name"].iloc[0] == best_model_name
    ).copy()
    best_test_predictions = best_test_predictions.drop(columns=["model_name"])
    return best_model_name, best_pipeline, benchmark_frame, best_test_predictions


def fit_calibrated_model(
    training_frame: pd.DataFrame,
    *,
    model_name: str,
    random_state: int,
    calibration_cv: int = 3,
    feature_columns: list[str] | None = None,
) -> CalibratedClassifierCV:
    if model_name == "elo_probability":
        raise ValueError("elo_probability is a deterministic baseline and cannot be calibrated via CalibratedClassifierCV.")
    resolved_feature_columns = feature_columns or FEATURE_COLUMNS
    pipelines = build_candidate_pipelines(random_state, feature_columns=resolved_feature_columns)
    base_pipeline = pipelines[model_name]
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=calibration_cv,
    )
    calibrated_model.fit(training_frame[resolved_feature_columns], training_frame["target"])
    return calibrated_model


def fit_full_model(
    training_frame: pd.DataFrame,
    *,
    model_name: str,
    random_state: int,
    feature_columns: list[str] | None = None,
) -> Pipeline:
    resolved_feature_columns = feature_columns or FEATURE_COLUMNS
    pipelines = build_candidate_pipelines(random_state, feature_columns=resolved_feature_columns)
    pipeline = pipelines[model_name]
    pipeline.fit(training_frame[resolved_feature_columns], training_frame["target"])
    return pipeline


def predict_frame_probabilities(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> pd.Series:
    resolved_feature_columns = feature_columns or FEATURE_COLUMNS
    if model == "elo_probability":
        return elo_expected_probability(frame["best_of_context_elo_gap"].fillna(frame["elo_gap"]).fillna(0.0))
    return pd.Series(model.predict_proba(frame[resolved_feature_columns])[:, 1], index=frame.index)


def model_insight_frame(
    pipeline: Any,
    *,
    sample_features: pd.DataFrame | None = None,
    sample_target: pd.Series | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_features is not None and sample_target is not None and not sample_features.empty:
        sample_size = min(len(sample_features), 5000)
        sampled_features = sample_features.sample(n=sample_size, random_state=random_state)
        sampled_target = sample_target.loc[sampled_features.index]
        importance = permutation_importance(
            pipeline,
            sampled_features,
            sampled_target,
            n_repeats=3,
            random_state=random_state,
            n_jobs=1,
        )
        frame = pd.DataFrame(
            {
                "feature": sampled_features.columns,
                "score_type": "permutation_importance",
                "score": importance.importances_mean,
            }
        )
        return frame.sort_values("score", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "score_type", "score"])


def save_artifacts(
    *,
    output_dir: str | Path,
    pipeline: Any,
    selected_model_name: str,
    benchmark_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    model_insights: pd.DataFrame,
    snapshot: pd.DataFrame,
    live_state: pd.DataFrame | None,
    pair_history: pd.DataFrame | None,
    summary_metrics: dict[str, Any],
    training_config: dict[str, Any],
    calibration_buckets: pd.DataFrame | None = None,
    error_slices: pd.DataFrame | None = None,
    high_confidence_misses: pd.DataFrame | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_path / "model.joblib")
    predictions.to_csv(output_path / "test_predictions.csv", index=False)
    benchmark_frame.to_csv(output_path / "model_benchmarks.csv", index=False)
    model_insights.to_csv(output_path / "feature_importances.csv", index=False)
    snapshot.to_csv(output_path / "player_snapshot.csv", index=False)
    if live_state is not None:
        live_state.to_csv(output_path / "player_live_state.csv", index=False)
    if pair_history is not None:
        pair_history.to_csv(output_path / "pair_history.csv", index=False)
    if calibration_buckets is not None:
        calibration_buckets.to_csv(output_path / "calibration_buckets.csv", index=False)
    if error_slices is not None:
        error_slices.to_csv(output_path / "error_slices.csv", index=False)
    if high_confidence_misses is not None:
        high_confidence_misses.to_csv(output_path / "high_confidence_misses.csv", index=False)

    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_metrics, handle, indent=2)

    with (output_path / "selected_model.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected_model": selected_model_name}, handle, indent=2)

    with (output_path / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(training_config, handle, indent=2)
