from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.bracket import simulate_tournament
from tennis_model.draws import fetch_atp_draw
from tennis_model.names import normalize_player_name
from tennis_model.predict import build_match_features, load_pair_history, load_snapshot, predict_match_probability_with_model


DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts_plus_stats_2026_model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "model.joblib"
DEFAULT_SNAPSHOT_PATH = DEFAULT_MODEL_DIR / "player_snapshot.csv"
DEFAULT_PAIR_HISTORY_PATH = DEFAULT_MODEL_DIR / "pair_history.csv"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"
DEFAULT_FEATURES_PATH = DEFAULT_MODEL_DIR / "feature_importances.csv"
DEFAULT_CALIBRATION_PATH = DEFAULT_MODEL_DIR / "calibration_buckets.csv"
DEFAULT_ERROR_SLICES_PATH = DEFAULT_MODEL_DIR / "error_slices.csv"
DEFAULT_HIGH_CONFIDENCE_MISSES_PATH = DEFAULT_MODEL_DIR / "high_confidence_misses.csv"
DEFAULT_PREDICTION_HISTORY_PATH = DEFAULT_MODEL_DIR / "prediction_history.csv"
DEFAULT_PREDICTION_HISTORY_BUCKETS_PATH = DEFAULT_MODEL_DIR / "prediction_history_buckets.csv"
DEFAULT_LIVE_RUN_DIR = PROJECT_ROOT / "artifacts_live" / "miami_main_auto_update_20260318_best"
DEFAULT_DRAW_RUN_DIR = PROJECT_ROOT / "artifacts_live" / "miami_main_draw_20260318_best"
DEFAULT_MATCHUPS_PATH = PROJECT_ROOT / "data" / "miami_open_2026_featured_matchups.csv"
DEFAULT_MATCH_SHEET_TEMPLATE = PROJECT_ROOT / "data" / "miami_open_2026_match_sheet_template.csv"
QUALIFYING_DATE = "2026-03-16"
MIAMI_DATE = "2026-03-18"
MIAMI_SURFACE = "Hard"
MIAMI_BEST_OF = 3
MIAMI_LEVEL = "M"
MIAMI_DRAW_SIZE = 96


def normalize_active_players(snapshot: pd.DataFrame) -> pd.DataFrame:
    active = snapshot.copy()
    active["last_match_date"] = pd.to_datetime(active["last_match_date"], errors="coerce")
    active["last_rank"] = pd.to_numeric(active["last_rank"], errors="coerce")
    active = active.loc[
        active["player_name"].str.len().gt(3)
        & (
            active["last_match_date"].ge(pd.Timestamp("2025-01-01"))
            | active["last_rank"].le(250)
        )
    ].copy()
    active = active.sort_values(["last_rank", "player_name"], na_position="last").reset_index(drop=True)
    return active


def z_score(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def contender_board(snapshot: pd.DataFrame) -> pd.DataFrame:
    board = normalize_active_players(snapshot).copy()
    board["miami_index"] = (
        0.45 * z_score(board["hard_elo"])
        + 0.20 * z_score(board["recent_elo"])
        + 0.15 * z_score(board["elo"])
        + 0.10 * z_score(board["hard_return_win_rate"])
        + 0.05 * z_score(board["hard_serve_win_rate"])
        + 0.05 * z_score(board["recent_return_win_rate"])
    )
    board["miami_index"] = 50 + 10 * board["miami_index"]
    return board.sort_values("miami_index", ascending=False).reset_index(drop=True)


@st.cache_resource
def load_model():
    import joblib

    return joblib.load(DEFAULT_MODEL_PATH)


@st.cache_data
def load_snapshot_frame() -> pd.DataFrame:
    return load_snapshot(DEFAULT_SNAPSHOT_PATH)


@st.cache_data
def load_pair_history_frame() -> pd.DataFrame | None:
    if DEFAULT_PAIR_HISTORY_PATH.exists():
        return load_pair_history(DEFAULT_PAIR_HISTORY_PATH)
    return None


@st.cache_data
def load_metrics() -> dict[str, float]:
    with DEFAULT_METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_feature_importances() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_FEATURES_PATH)


@st.cache_data
def load_calibration_buckets() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_CALIBRATION_PATH)


@st.cache_data
def load_error_slices() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_ERROR_SLICES_PATH)


@st.cache_data
def load_high_confidence_misses() -> pd.DataFrame:
    if DEFAULT_HIGH_CONFIDENCE_MISSES_PATH.exists():
        return pd.read_csv(DEFAULT_HIGH_CONFIDENCE_MISSES_PATH)
    return pd.DataFrame()


@st.cache_data
def load_prediction_history() -> pd.DataFrame:
    if DEFAULT_PREDICTION_HISTORY_PATH.exists():
        return pd.read_csv(DEFAULT_PREDICTION_HISTORY_PATH)
    return pd.DataFrame()


@st.cache_data
def load_prediction_history_buckets() -> pd.DataFrame:
    if DEFAULT_PREDICTION_HISTORY_BUCKETS_PATH.exists():
        return pd.read_csv(DEFAULT_PREDICTION_HISTORY_BUCKETS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_featured_matchups() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_MATCHUPS_PATH)


def available_run_dirs() -> list[Path]:
    runs_dir = PROJECT_ROOT / "artifacts_live"
    if not runs_dir.exists():
        return []
    return sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def run_display_name(path: Path) -> str:
    return path.name.replace("_", " ")


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_run_bundle(run_dir: Path) -> dict[str, pd.DataFrame | Path]:
    return {
        "run_dir": run_dir,
        "current_round_predictions": load_optional_csv(run_dir / "current_round_predictions.csv"),
        "current_round_bracket": load_optional_csv(run_dir / "current_round_bracket.csv"),
        "completed_match_audit": load_optional_csv(run_dir / "completed_match_audit.csv"),
        "completed_results_auto": load_optional_csv(run_dir / "completed_results_auto.csv"),
        "remaining_tournament_probabilities": load_optional_csv(run_dir / "remaining_tournament_probabilities.csv"),
        "first_round_predictions": load_optional_csv(run_dir / "first_round_predictions.csv"),
    }


def run_summary_text(bundle: dict[str, pd.DataFrame | Path]) -> str:
    completed = bundle["completed_match_audit"]
    upcoming = bundle["current_round_predictions"]
    if not completed.empty:
        latest_date = completed["match_date"].dropna().astype(str).max() if "match_date" in completed.columns else "unknown"
        accuracy = completed["correct_pick"].mean() if "correct_pick" in completed.columns else float("nan")
        return f"{len(completed)} completed matches scored through {latest_date} | run accuracy {accuracy:.3f}"
    if not upcoming.empty:
        return f"{len(upcoming)} upcoming matches currently priced"
    first_round = bundle["first_round_predictions"]
    if not first_round.empty:
        return f"{len(first_round)} opening-round matches priced"
    return "No scored matches found in this run yet"


def confidence_label(probability: float) -> str:
    edge = abs(probability - 0.5)
    if edge < 0.05:
        return "Coin Flip"
    if edge < 0.10:
        return "Lean"
    if edge < 0.20:
        return "Strong"
    return "Very Strong"


def simplify_upcoming_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    simplified = frame.copy()
    simplified["match"] = simplified["player_1"] + " vs " + simplified["player_2"]
    simplified["favorite_win_probability"] = simplified.apply(
        lambda row: row["player_1_win_probability"] if row["favorite"] == row["player_1"] else 1 - row["player_1_win_probability"],
        axis=1,
    )
    simplified["confidence_band"] = simplified["favorite_win_probability"].map(confidence_label)
    columns = ["match"]
    if "round_name" in simplified.columns:
        columns.append("round_name")
    columns.extend(["favorite", "favorite_win_probability", "confidence_band"])
    return simplified[columns].sort_values(
        ["round_name", "favorite_win_probability"] if "round_name" in simplified.columns else ["favorite_win_probability"],
        ascending=[True, False] if "round_name" in simplified.columns else [False],
    ).reset_index(drop=True)


def simplify_completed_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    simplified = frame.copy()
    simplified["match"] = simplified["player_1"] + " vs " + simplified["player_2"]
    simplified["favorite_win_probability"] = simplified["favorite_win_probability"].astype(float)
    simplified["confidence_band"] = simplified["favorite_win_probability"].map(confidence_label)
    simplified["result"] = simplified["correct_pick"].map({True: "Correct", False: "Wrong"})
    return simplified[
        [
            "match_date",
            "round_name",
            "match",
            "favorite",
            "winner",
            "favorite_win_probability",
            "confidence_band",
            "result",
        ]
    ].sort_values(["match_date", "round_name", "favorite_win_probability"], ascending=[False, True, False]).reset_index(drop=True)


def tracker_probability_chart(frame: pd.DataFrame, *, title: str) -> go.Figure:
    chart_frame = frame.head(20).copy()
    chart_frame["win_pct"] = 100 * chart_frame["favorite_win_probability"]
    figure = px.bar(
        chart_frame.sort_values("win_pct", ascending=True),
        x="win_pct",
        y="match",
        orientation="h",
        color="favorite",
        labels={"win_pct": "Favorite Win %", "match": ""},
        title=title,
    )
    figure.update_layout(height=max(360, 28 * len(chart_frame)), margin=dict(l=10, r=10, t=50, b=10))
    return figure


def prediction_probability(
    model,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    *,
    player_1: str,
    player_2: str,
    round_name: str,
    match_date: str,
) -> float:
    return predict_match_probability_with_model(
        model,
        snapshot,
        player_1=player_1,
        player_2=player_2,
        match_date=match_date,
        surface=MIAMI_SURFACE,
        best_of=MIAMI_BEST_OF,
        round_name=round_name,
        tourney_level=MIAMI_LEVEL,
        draw_size=MIAMI_DRAW_SIZE,
        pair_history=pair_history,
    )


def player_profile(snapshot: pd.DataFrame, player_name: str) -> dict[str, float]:
    row = snapshot.loc[snapshot["player_name"] == player_name].iloc[0]
    return {
        "Hard Elo": float(row["hard_elo"]),
        "Recent Elo": float(row["recent_elo"]),
        "Overall Elo": float(row["elo"]),
        "Hard Serve Win %": 100 * float(row["hard_serve_win_rate"]),
        "Hard Return Win %": 100 * float(row["hard_return_win_rate"]),
        "Recent Return Win %": 100 * float(row["recent_return_win_rate"]),
    }


def matchup_radar(snapshot: pd.DataFrame, player_1: str, player_2: str) -> go.Figure:
    profile_1 = player_profile(snapshot, player_1)
    profile_2 = player_profile(snapshot, player_2)
    categories = list(profile_1.keys())

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=list(profile_1.values()),
            theta=categories,
            fill="toself",
            name=player_1,
            line_color="#ff6b35",
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=list(profile_2.values()),
            theta=categories,
            fill="toself",
            name=player_2,
            line_color="#1d4ed8",
        )
    )
    figure.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
        height=420,
    )
    return figure


def probability_gauge(player_1: str, probability: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": f"{player_1} Win Chance"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6b35"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    figure.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return figure


def matchup_driver_frame(
    model,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    feature_importances: pd.DataFrame,
    *,
    player_1: str,
    player_2: str,
    match_date: str,
    round_name: str,
    best_of: int,
    draw_size: int,
) -> pd.DataFrame:
    feature_row = build_match_features(
        snapshot,
        player_1=player_1,
        player_2=player_2,
        match_date=pd.Timestamp(match_date),
        surface=MIAMI_SURFACE,
        best_of=best_of,
        round_name=round_name,
        tourney_level=MIAMI_LEVEL,
        draw_size=draw_size,
        pair_history=pair_history,
    )
    baseline_probability = float(model.predict_proba(feature_row)[:, 1][0])
    importance_map = dict(zip(feature_importances["feature"], feature_importances["score"]))

    rows: list[dict[str, object]] = []
    for feature in feature_row.columns:
        if feature in {"best_of", "draw_size", "surface", "round", "tourney_level"}:
            continue
        value = feature_row.iloc[0][feature]
        if pd.isna(value):
            continue
        neutralized = feature_row.copy()
        neutralized.at[0, feature] = 0
        neutral_probability = float(model.predict_proba(neutralized)[:, 1][0])
        impact = baseline_probability - neutral_probability
        rows.append(
            {
                "feature": feature,
                "impact": impact,
                "absolute_impact": abs(impact),
                "driver": player_1 if impact >= 0 else player_2,
                "global_importance": float(importance_map.get(feature, 0.0)),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["absolute_impact", "global_importance"], ascending=[False, False]).reset_index(drop=True)


def matchup_driver_chart(driver_frame: pd.DataFrame) -> go.Figure:
    top_drivers = driver_frame.head(10).sort_values("impact", ascending=True)
    figure = px.bar(
        top_drivers,
        x="impact",
        y="feature",
        orientation="h",
        color="driver",
        labels={"impact": "Probability Push", "feature": "", "driver": "Pushes Toward"},
        color_discrete_sequence=["#ff6b35", "#1d4ed8"],
    )
    figure.add_vline(x=0, line_width=1, line_dash="dash", line_color="#6b7280")
    figure.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    return figure


def featured_prediction_table(model, snapshot: pd.DataFrame, pair_history: pd.DataFrame | None) -> pd.DataFrame:
    matchups = load_featured_matchups().copy()
    matchups["player_1_win_probability"] = matchups.apply(
        lambda row: prediction_probability(
            model,
            snapshot,
            pair_history,
            player_1=row["player_1"],
            player_2=row["player_2"],
            round_name=row["round_name"],
            match_date=MIAMI_DATE,
        ),
        axis=1,
    )
    matchups["favorite"] = matchups.apply(
        lambda row: row["player_1"]
        if row["player_1_win_probability"] >= 0.5
        else row["player_2"],
        axis=1,
    )
    matchups["edge"] = (matchups["player_1_win_probability"] - 0.5).abs() * 200
    return matchups


def score_match_sheet(model, snapshot: pd.DataFrame, pair_history: pd.DataFrame | None, match_sheet: pd.DataFrame) -> pd.DataFrame:
    frame = match_sheet.copy()
    for required in ["player_1", "player_2"]:
        if required not in frame.columns:
            raise ValueError("Match sheet must include player_1 and player_2 columns.")

    if "round_name" not in frame.columns:
        frame["round_name"] = "Q1"
    if "match_date" not in frame.columns:
        frame["match_date"] = QUALIFYING_DATE
    if "surface" not in frame.columns:
        frame["surface"] = MIAMI_SURFACE
    if "best_of" not in frame.columns:
        frame["best_of"] = MIAMI_BEST_OF
    if "tourney_level" not in frame.columns:
        frame["tourney_level"] = MIAMI_LEVEL
    if "draw_size" not in frame.columns:
        frame["draw_size"] = 48

    frame["player_1_win_probability"] = frame.apply(
        lambda row: predict_match_probability_with_model(
            model,
            snapshot,
            player_1=row["player_1"],
            player_2=row["player_2"],
            match_date=str(row["match_date"]),
            surface=str(row["surface"]),
            best_of=int(row["best_of"]),
            round_name=str(row["round_name"]),
            tourney_level=str(row["tourney_level"]),
            draw_size=int(row["draw_size"]),
            pair_history=pair_history,
        ),
        axis=1,
    )
    frame["favorite"] = frame.apply(
        lambda row: row["player_1"] if row["player_1_win_probability"] >= 0.5 else row["player_2"],
        axis=1,
    )
    frame["confidence"] = (frame["player_1_win_probability"] - 0.5).abs() * 200
    return frame


def simulate_uploaded_bracket(
    model_path: Path,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    bracket_frame: pd.DataFrame,
    simulations: int,
) -> pd.DataFrame:
    pairs = list(bracket_frame[["player_1", "player_2"]].itertuples(index=False, name=None))
    return simulate_tournament(
        model_path=model_path,
        snapshot=snapshot,
        pair_history=pair_history,
        first_round_pairs=pairs,
        tournament_date=MIAMI_DATE,
        surface=MIAMI_SURFACE,
        best_of=MIAMI_BEST_OF,
        tourney_level=MIAMI_LEVEL,
        draw_size=len(pairs) * 2,
        simulations=simulations,
    )


def score_draw_pairs(
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
                "confidence": abs(probability - 0.5) * 200 if probability is not None else None,
                "model_ready": model_ready,
                "player_1_in_snapshot": player_1_in_snapshot,
                "player_2_in_snapshot": player_2_in_snapshot,
            }
        )
    return pd.DataFrame(rows)


def top_contender_chart(board: pd.DataFrame) -> go.Figure:
    top_board = board.head(12).sort_values("miami_index", ascending=True)
    figure = px.bar(
        top_board,
        x="miami_index",
        y="player_name",
        orientation="h",
        color="hard_elo",
        color_continuous_scale=["#fde68a", "#f97316", "#7c2d12"],
        labels={"miami_index": "Miami Index", "player_name": ""},
    )
    figure.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
    return figure


def feature_chart(feature_frame: pd.DataFrame) -> go.Figure:
    top_features = feature_frame.head(12).sort_values("score", ascending=True)
    figure = px.bar(
        top_features,
        x="score",
        y="feature",
        orientation="h",
        color="score",
        color_continuous_scale=["#93c5fd", "#2563eb"],
        labels={"score": "Importance", "feature": ""},
    )
    figure.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
    return figure


def probability_breakdown_chart(featured_predictions: pd.DataFrame) -> go.Figure:
    chart_frame = featured_predictions.copy()
    chart_frame["player_1_pct"] = 100 * chart_frame["player_1_win_probability"]
    figure = px.bar(
        chart_frame,
        x="label",
        y="player_1_pct",
        color="favorite",
        text=chart_frame["player_1_pct"].map(lambda value: f"{value:.1f}%"),
        labels={"label": "", "player_1_pct": "Player 1 Win %"},
    )
    figure.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return figure


def calibration_chart(calibration_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="#9ca3af", dash="dash"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=calibration_frame["avg_predicted_probability"],
            y=calibration_frame["actual_win_rate"],
            mode="lines+markers",
            name="Model",
            line=dict(color="#ea580c", width=3),
            marker=dict(size=10),
            text=calibration_frame["matches"].map(lambda value: f"{value} matches"),
        )
    )
    figure.update_layout(
        xaxis_title="Average Predicted Probability",
        yaxis_title="Actual Win Rate",
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return figure


def error_slice_chart(error_slices: pd.DataFrame) -> go.Figure:
    round_slices = error_slices.loc[error_slices["slice_name"] == "round"].copy()
    round_slices["label"] = round_slices["round"].fillna("Unknown")
    figure = px.bar(
        round_slices.sort_values("accuracy"),
        x="accuracy",
        y="label",
        orientation="h",
        color="matches",
        color_continuous_scale=["#bfdbfe", "#2563eb"],
        labels={"accuracy": "Accuracy", "label": "Round"},
    )
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    return figure


def completed_match_review_chart(audit_frame: pd.DataFrame) -> go.Figure:
    chart_frame = audit_frame.copy()
    chart_frame["match_label"] = chart_frame["player_1"] + " vs " + chart_frame["player_2"]
    chart_frame["favorite_win_pct"] = 100 * chart_frame["favorite_win_probability"]
    chart_frame["result_label"] = chart_frame["correct_pick"].map({True: "Correct", False: "Wrong"})
    top_frame = chart_frame.sort_values(["correct_pick", "confidence"], ascending=[True, False]).head(20)
    figure = px.bar(
        top_frame.sort_values("favorite_win_pct", ascending=True),
        x="favorite_win_pct",
        y="match_label",
        orientation="h",
        color="result_label",
        hover_data=["winner", "favorite", "round_name"],
        labels={"favorite_win_pct": "Favorite Win %", "match_label": ""},
        color_discrete_map={"Correct": "#16a34a", "Wrong": "#dc2626"},
    )
    figure.update_layout(height=max(320, 28 * len(top_frame)), margin=dict(l=10, r=10, t=30, b=10))
    return figure


def high_confidence_miss_chart(miss_frame: pd.DataFrame) -> go.Figure:
    chart_frame = miss_frame.copy()
    chart_frame["match_label"] = chart_frame["player_1"] + " vs " + chart_frame["player_2"]
    chart_frame["favorite_win_pct"] = 100 * chart_frame["favorite_win_probability"]
    figure = px.bar(
        chart_frame.sort_values("favorite_win_pct", ascending=True),
        x="favorite_win_pct",
        y="match_label",
        orientation="h",
        color="favorite",
        hover_data=["actual_winner", "round", "surface"],
        labels={"favorite_win_pct": "Favorite Win %", "match_label": ""},
        color_discrete_sequence=["#dc2626"],
    )
    figure.update_layout(height=max(320, 28 * len(chart_frame)), margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Miami Open Predictor",
        page_icon="🎾",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .hero-card {
            padding: 1.2rem 1.3rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #fff3e6 0%, #ffffff 100%);
            border: 1px solid rgba(249, 115, 22, 0.18);
        }
        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    snapshot = load_snapshot_frame()
    pair_history = load_pair_history_frame()
    metrics = load_metrics()
    feature_importances = load_feature_importances()
    calibration_buckets = load_calibration_buckets()
    error_slices = load_error_slices()
    high_confidence_misses = load_high_confidence_misses()
    prediction_history = load_prediction_history()
    prediction_history_buckets = load_prediction_history_buckets()
    board = contender_board(snapshot)
    active_players = board["player_name"].tolist()
    featured_predictions = featured_prediction_table(model, snapshot, pair_history)
    run_dirs = available_run_dirs()
    default_run_dir = run_dirs[0] if run_dirs else None
    if DEFAULT_LIVE_RUN_DIR.exists() and DEFAULT_LIVE_RUN_DIR in run_dirs:
        default_run_dir = run_dirs[0]
    default_run_index = run_dirs.index(default_run_dir) if default_run_dir in run_dirs else 0

    st.sidebar.header("Tracker View")
    selected_run_dir = None
    if run_dirs:
        selected_run_dir = st.sidebar.selectbox(
            "Prediction Run",
            run_dirs,
            index=default_run_index,
            format_func=run_display_name,
        )
    tracker_bundle = load_run_bundle(selected_run_dir) if selected_run_dir is not None else {
        "current_round_predictions": pd.DataFrame(),
        "current_round_bracket": pd.DataFrame(),
        "completed_match_audit": pd.DataFrame(),
        "completed_results_auto": pd.DataFrame(),
        "remaining_tournament_probabilities": pd.DataFrame(),
        "first_round_predictions": pd.DataFrame(),
    }
    if selected_run_dir is not None:
        st.sidebar.caption(run_summary_text(tracker_bundle))
    upcoming_predictions = tracker_bundle["current_round_predictions"]
    if upcoming_predictions.empty:
        fallback_draw_dir = DEFAULT_DRAW_RUN_DIR if DEFAULT_DRAW_RUN_DIR.exists() else selected_run_dir
        if fallback_draw_dir is not None:
            draw_bundle = load_run_bundle(fallback_draw_dir)
            upcoming_predictions = draw_bundle["first_round_predictions"]
    simplified_upcoming = simplify_upcoming_predictions(upcoming_predictions)
    simplified_completed = simplify_completed_predictions(tracker_bundle["completed_match_audit"])

    st.markdown("## Miami Open ATP Dashboard")
    st.markdown(
        """
        <div class="hero-card">
        <h4 style="margin: 0 0 0.4rem 0;">Miami Open 2026 Live Tracker</h4>
        <div>Qualifying is underway now, so this dashboard is ready for live qualifying match predictions today and full main-draw simulation as soon as the bracket is posted.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Calibrated Test Accuracy", f"{metrics['calibrated_test_accuracy']:.3f}")
    metric_cols[1].metric("Calibrated Test AUC", f"{metrics['calibrated_test_roc_auc']:.3f}")
    metric_cols[2].metric("Calibrated Log Loss", f"{metrics['calibrated_test_log_loss']:.3f}")
    metric_cols[3].metric("Selected Model", metrics["model_name"].replace("_", " ").title())

    st.subheader("Simple Prediction Tracker")
    tracker_cols = st.columns(4)
    tracker_cols[0].metric("Upcoming Matches", len(simplified_upcoming))
    tracker_cols[1].metric("Completed In Run", len(simplified_completed))
    tracker_cols[2].metric(
        "Run Accuracy",
        f"{tracker_bundle['completed_match_audit']['correct_pick'].mean():.3f}" if not tracker_bundle["completed_match_audit"].empty else "N/A",
    )
    tracker_cols[3].metric(
        "History Accuracy",
        f"{prediction_history['correct_pick'].mean():.3f}" if not prediction_history.empty else "N/A",
    )
    if selected_run_dir is not None:
        st.caption(f"Viewing run folder: `{selected_run_dir.name}`")

    tracker_tab_upcoming, tracker_tab_completed, tracker_tab_history = st.tabs(
        ["Upcoming Predictions", "Finished Matches", "Long-Run History"]
    )

    with tracker_tab_upcoming:
        if not simplified_upcoming.empty:
            st.plotly_chart(
                tracker_probability_chart(simplified_upcoming, title="Upcoming Match Predictions"),
                width="stretch",
                key="tracker_upcoming_chart",
            )
            st.dataframe(
                simplified_upcoming.assign(
                    favorite_win_probability=lambda frame: frame["favorite_win_probability"].map(lambda value: f"{value:.3f}")
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No upcoming prediction file found in the selected run yet.")

    with tracker_tab_completed:
        if not simplified_completed.empty:
            wrong_count = int((tracker_bundle["completed_match_audit"]["correct_pick"] == False).sum())
            st.metric("Wrong Picks In Run", wrong_count)
            st.plotly_chart(
                completed_match_review_chart(tracker_bundle["completed_match_audit"]),
                width="stretch",
                key="tracker_completed_chart",
            )
            st.dataframe(
                simplified_completed.assign(
                    favorite_win_probability=lambda frame: frame["favorite_win_probability"].map(lambda value: f"{value:.3f}")
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No completed match audit is available for this run yet.")

    with tracker_tab_history:
        if not prediction_history.empty:
            if not prediction_history_buckets.empty:
                st.plotly_chart(
                    calibration_chart(prediction_history_buckets),
                    width="stretch",
                    key="tracker_history_calibration_chart",
                )
            history_view = prediction_history[
                [
                    "match_date",
                    "round_name",
                    "player_1",
                    "player_2",
                    "winner",
                    "favorite",
                    "favorite_win_probability",
                    "correct_pick",
                ]
            ].copy()
            history_view["match"] = history_view["player_1"] + " vs " + history_view["player_2"]
            st.dataframe(
                history_view[
                    [
                        "match_date",
                        "round_name",
                        "match",
                        "winner",
                        "favorite",
                        "favorite_win_probability",
                        "correct_pick",
                    ]
                ]
                .sort_values(["match_date", "round_name"], ascending=[False, False])
                .head(50)
                .assign(
                    favorite_win_probability=lambda frame: frame["favorite_win_probability"].map(lambda value: f"{value:.3f}")
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No long-run prediction history has been logged yet.")

    left_col, right_col = st.columns([1.2, 1.0])
    with left_col:
        st.subheader("Top Miami Contenders")
        st.plotly_chart(top_contender_chart(board), width="stretch", key="top_contender_chart")
        st.caption("Miami Index is a model-facing hard-court strength blend of hard Elo, recent Elo, overall Elo, and recent/surface return and serve form.")

    with right_col:
        st.subheader("What The Model Uses Most")
        st.plotly_chart(feature_chart(feature_importances), width="stretch", key="feature_importance_chart")

    diag_left, diag_right = st.columns(2)
    with diag_left:
        st.subheader("Calibration Check")
        st.plotly_chart(calibration_chart(calibration_buckets), width="stretch", key="model_calibration_chart")
        st.caption("If the model says 70%, we want players in that bucket to win about 70% of the time.")
    with diag_right:
        st.subheader("Where The Model Struggles")
        st.plotly_chart(error_slice_chart(error_slices), width="stretch", key="error_slice_chart")
        st.caption("This slice view helps us see whether the model is weaker in certain rounds or match contexts.")

    st.subheader("High-Confidence Misses")
    if not high_confidence_misses.empty:
        st.plotly_chart(
            high_confidence_miss_chart(high_confidence_misses.head(15)),
            width="stretch",
            key="high_confidence_miss_chart",
        )
        st.dataframe(
            high_confidence_misses.head(15).assign(
                favorite_win_probability=lambda frame: frame["favorite_win_probability"].map(lambda value: f"{value:.3f}"),
                confidence=lambda frame: frame["confidence"].map(lambda value: f"{value:.1f}"),
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("These are the toughest wrong calls from the latest holdout test set, which is a very useful place to look for missing features.")
    else:
        st.caption("No high-confidence misses file found yet. Retrain the model to generate it.")

    st.subheader("Featured Miami Matchups")
    st.plotly_chart(probability_breakdown_chart(featured_predictions), width="stretch", key="featured_matchups_chart")
    st.dataframe(
        featured_predictions[
            ["label", "player_1", "player_2", "round_name", "favorite", "player_1_win_probability"]
        ].assign(
            player_1_win_probability=lambda frame: frame["player_1_win_probability"].map(lambda value: f"{value:.3f}")
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("These are featured possible Miami matchups, not the official current draw.")

    st.subheader("Live Qualifying Match Sheet")
    st.markdown(
        "Upload today's qualifying pairings or any Miami order-of-play style sheet. "
        "Use `data/miami_open_2026_match_sheet_template.csv` as a template."
    )
    match_sheet_upload = st.file_uploader("Qualifying Match Sheet CSV", type="csv", key="match_sheet")
    if match_sheet_upload is not None:
        try:
            match_sheet = pd.read_csv(match_sheet_upload)
            scored_sheet = score_match_sheet(model, snapshot, pair_history, match_sheet)
            st.dataframe(
                scored_sheet[
                    [
                        "player_1",
                        "player_2",
                        "round_name",
                        "match_date",
                        "favorite",
                        "player_1_win_probability",
                        "confidence",
                    ]
                ].assign(
                    player_1_win_probability=lambda frame: frame["player_1_win_probability"].map(lambda value: f"{value:.3f}"),
                    confidence=lambda frame: frame["confidence"].map(lambda value: f"{value:.1f}"),
                ),
                use_container_width=True,
                hide_index=True,
            )
            live_chart = px.bar(
                scored_sheet,
                x="player_1_win_probability",
                y=scored_sheet.apply(lambda row: f"{row['player_1']} vs {row['player_2']}", axis=1),
                orientation="h",
                color="favorite",
                labels={"player_1_win_probability": "Player 1 Win Probability", "y": ""},
            )
            live_chart.update_layout(height=max(320, 70 * len(scored_sheet)), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(live_chart, width="stretch", key="live_match_sheet_chart")
        except Exception as exc:
            st.error(f"Could not score the uploaded match sheet: {exc}")
    else:
        st.markdown(
            '<div class="small-note">No live qualifying sheet uploaded yet. The app is ready for any CSV with `player_1` and `player_2`, plus optional round/date/context columns.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Quick Match Predictor")
    select_cols = st.columns(2)
    default_p1 = active_players.index("Carlos Alcaraz") if "Carlos Alcaraz" in active_players else 0
    default_p2 = active_players.index("Jannik Sinner") if "Jannik Sinner" in active_players else min(1, len(active_players) - 1)
    player_1 = select_cols[0].selectbox("Player 1", active_players, index=default_p1)
    player_2 = select_cols[1].selectbox("Player 2", active_players, index=default_p2)

    context_cols = st.columns(4)
    match_date = context_cols[0].date_input("Match Date", value=pd.Timestamp(QUALIFYING_DATE))
    round_name = context_cols[1].selectbox("Round", ["Q1", "Q2", "Q3", "R128", "R64", "R32", "R16", "QF", "SF", "F"], index=0)
    best_of = context_cols[2].selectbox("Best Of", [3, 5], index=0)
    draw_size = context_cols[3].selectbox("Draw Size", [48, 96, 128, 64, 32], index=0)

    if player_1 == player_2:
        st.warning("Choose two different players to generate a prediction.")
    else:
        probability = predict_match_probability_with_model(
            model,
            snapshot,
            player_1=player_1,
            player_2=player_2,
            match_date=str(match_date),
            surface=MIAMI_SURFACE,
            best_of=int(best_of),
            round_name=round_name,
            tourney_level=MIAMI_LEVEL,
            draw_size=int(draw_size),
            pair_history=pair_history,
        )
        result_cols = st.columns([0.8, 1.2])
        with result_cols[0]:
            st.plotly_chart(probability_gauge(player_1, probability), width="stretch", key="quick_predictor_gauge")
            favorite = player_1 if probability >= 0.5 else player_2
            st.markdown(f"**Favorite:** {favorite}")
            st.markdown(f"**{player_2} win chance:** {1 - probability:.3f}")
        with result_cols[1]:
            st.plotly_chart(matchup_radar(snapshot, player_1, player_2), width="stretch", key="quick_predictor_radar")
            driver_frame = matchup_driver_frame(
                model,
                snapshot,
                pair_history,
                feature_importances,
                player_1=player_1,
                player_2=player_2,
                match_date=str(match_date),
                round_name=round_name,
                best_of=int(best_of),
                draw_size=int(draw_size),
            )
            if not driver_frame.empty:
                st.plotly_chart(matchup_driver_chart(driver_frame), width="stretch", key="quick_predictor_driver_chart")
                st.caption("This is a local sensitivity view: each bar shows how much that feature is pushing the match probability toward one player or the other.")

    st.subheader("Bracket Simulator")
    st.markdown(
        "Upload a bracket CSV with `player_1` and `player_2` columns. "
        "For Miami's 96-player format, include `BYE` in the empty seeded slots so the simulator can advance those players automatically. "
        "Use the template at `data/miami_open_2026_bracket_template.csv` when the draw is published."
    )
    uploaded = st.file_uploader("Miami bracket CSV", type="csv")
    simulations = st.slider("Simulations", min_value=200, max_value=5000, value=1000, step=200)

    if uploaded is not None:
        bracket_frame = pd.read_csv(uploaded)
        if not {"player_1", "player_2"}.issubset(bracket_frame.columns):
            st.error("Your CSV needs `player_1` and `player_2` columns.")
        else:
            bracket_results = simulate_uploaded_bracket(
                DEFAULT_MODEL_PATH,
                snapshot,
                pair_history,
                bracket_frame,
                simulations=simulations,
            )
            st.dataframe(bracket_results.head(16), use_container_width=True, hide_index=True)
            champion_chart = px.bar(
                bracket_results.head(12).sort_values("Champion", ascending=True),
                x="Champion",
                y="player_name",
                orientation="h",
                color="Champion",
                color_continuous_scale=["#fed7aa", "#ea580c"],
                labels={"Champion": "Title Probability", "player_name": ""},
            )
            champion_chart.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
            st.plotly_chart(champion_chart, width="stretch", key="uploaded_bracket_champion_chart")
    else:
        st.markdown(
            '<div class="small-note">No official Miami draw loaded yet. The upload section is ready for the moment the bracket is public.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Prediction History")
    if not prediction_history.empty:
        history_cols = st.columns(4)
        history_cols[0].metric("Logged Matches", len(prediction_history))
        history_cols[1].metric("Logged Accuracy", f"{prediction_history['correct_pick'].mean():.3f}")
        history_cols[2].metric("Logged Avg Log Loss", f"{prediction_history['log_loss'].mean():.3f}")
        history_cols[3].metric("Logged Avg Brier", f"{prediction_history['brier_error'].mean():.3f}")
        if not prediction_history_buckets.empty:
            st.plotly_chart(
                calibration_chart(prediction_history_buckets),
                width="stretch",
                key="prediction_history_calibration_chart",
            )
        st.dataframe(
            prediction_history[
                [
                    "match_date",
                    "round_name",
                    "player_1",
                    "player_2",
                    "winner",
                    "favorite",
                    "player_1_win_probability",
                    "correct_pick",
                ]
            ].tail(30),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("This file updates as live tournament runs finish and gives you a reusable record of predicted probabilities versus actual outcomes.")
    else:
        st.caption("No persistent prediction history yet. Run the auto updater to start building it.")

    st.subheader("Completed Match Review")
    st.markdown(
        "Upload `completed_match_audit.csv` from the live updater to see which finished matches the model got right or wrong."
    )
    completed_audit_upload = st.file_uploader("Completed Match Audit CSV", type="csv", key="completed_audit")
    if completed_audit_upload is not None:
        try:
            completed_audit = pd.read_csv(completed_audit_upload)
            review_cols = st.columns(4)
            review_cols[0].metric("Matches Scored", len(completed_audit))
            review_cols[1].metric("Accuracy", f"{completed_audit['correct_pick'].mean():.3f}")
            review_cols[2].metric("Avg Log Loss", f"{completed_audit['log_loss'].mean():.3f}")
            review_cols[3].metric("Avg Brier", f"{completed_audit['brier_error'].mean():.3f}")
            st.plotly_chart(
                completed_match_review_chart(completed_audit),
                width="stretch",
                key="uploaded_completed_match_review_chart",
            )
            st.dataframe(
                completed_audit[
                    [
                        "match_date",
                        "round_name",
                        "player_1",
                        "player_2",
                        "winner",
                        "favorite",
                        "favorite_win_probability",
                        "correct_pick",
                        "confidence",
                    ]
                ].assign(
                    favorite_win_probability=lambda frame: frame["favorite_win_probability"].map(lambda value: f"{value:.3f}"),
                    confidence=lambda frame: frame["confidence"].map(lambda value: f"{value:.1f}"),
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Could not read the completed match audit file: {exc}")
    else:
        st.markdown(
            '<div class="small-note">Run the auto updater first, then upload the generated `completed_match_audit.csv` here for a live scorecard.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Official ATP Draw Import")
    st.markdown(
        "Paste an ATP draw page URL and the app will pull the official bracket, score the first round, "
        "and simulate the event repeatedly for round-by-round probabilities."
    )
    draw_cols = st.columns([1.7, 0.6, 0.6, 0.6, 0.6])
    draw_url = draw_cols[0].text_input(
        "ATP Draw URL",
        value="https://www.atptour.com/en/scores/current/miami/403/draws",
    )
    draw_match_type = draw_cols[1].selectbox("Match Type", ["singles", "qualifiersingles"], index=1)
    draw_surface = draw_cols[2].selectbox("Surface", ["Hard", "Clay", "Grass"], index=0)
    draw_best_of = draw_cols[3].selectbox("Best Of", [3, 5], index=0)
    draw_level = draw_cols[4].selectbox("Level", ["M", "G", "A", "F", "O"], index=0)
    official_simulations = st.slider(
        "Official Draw Simulations",
        min_value=200,
        max_value=5000,
        value=1000,
        step=200,
    )
    allow_unresolved = st.checkbox(
        "Allow unresolved placeholder slots",
        value=False,
        help="Turn this on only if the draw still contains unresolved Qualifier/TBA spots and you still want a provisional simulation.",
    )

    if st.button("Fetch And Simulate Official Draw", use_container_width=True):
        try:
            draw = fetch_atp_draw(draw_url, match_type=draw_match_type)
            tournament_date = draw.tournament_date or str(MIAMI_DATE)
            level = draw.inferred_tourney_level or draw_level
            best_of = draw.inferred_best_of or int(draw_best_of)
            scored_draw = score_draw_pairs(
                model,
                snapshot,
                pair_history,
                draw.first_round_pairs,
                placeholder_names=set(draw.placeholder_names),
                match_date=tournament_date,
                surface=draw_surface,
                best_of=int(best_of),
                round_name=draw.first_round_label,
                tourney_level=level,
                draw_size=draw.draw_size,
            )

            info_cols = st.columns(5)
            info_cols[0].metric("Tournament", draw.tournament_title)
            info_cols[1].metric("Match Type", draw.match_type)
            info_cols[2].metric("Draw Size", draw.draw_size)
            info_cols[3].metric("Start Date", tournament_date or "Unknown")
            info_cols[4].metric("Terminal Outcome", draw.terminal_label)

            st.caption(
                f"Rounds: {', '.join(draw.round_labels)} | Source: {draw.source_url}"
            )

            st.dataframe(
                scored_draw[
                    [
                        "player_1",
                        "player_2",
                        "favorite",
                        "player_1_win_probability",
                        "confidence",
                        "player_1_in_snapshot",
                        "player_2_in_snapshot",
                    ]
                ].assign(
                    player_1_win_probability=lambda frame: frame["player_1_win_probability"].map(lambda value: f"{value:.3f}"),
                    confidence=lambda frame: frame["confidence"].map(lambda value: f"{value:.1f}"),
                ),
                use_container_width=True,
                hide_index=True,
            )

            unresolved = scored_draw.loc[~scored_draw["model_ready"]]
            if not unresolved.empty and not allow_unresolved:
                st.warning(
                    "The official draw still has unresolved placeholder slots. "
                    "First-round probabilities are shown, but tournament simulation is paused until those slots are filled."
                )
            else:
                official_probabilities = simulate_tournament(
                    model_path=DEFAULT_MODEL_PATH,
                    snapshot=snapshot,
                    pair_history=pair_history,
                    first_round_pairs=draw.first_round_pairs,
                    tournament_date=tournament_date,
                    surface=draw_surface,
                    best_of=int(best_of),
                    tourney_level=level,
                    draw_size=draw.draw_size,
                    simulations=official_simulations,
                    round_labels=draw.round_labels,
                    terminal_label=draw.terminal_label,
                )
                terminal_chart = px.bar(
                    official_probabilities.head(16).sort_values(draw.terminal_label, ascending=True),
                    x=draw.terminal_label,
                    y="player_name",
                    orientation="h",
                    color=draw.terminal_label,
                    color_continuous_scale=["#fed7aa", "#ea580c"],
                    labels={draw.terminal_label: f"{draw.terminal_label} Probability", "player_name": ""},
                )
                terminal_chart.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=30, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(terminal_chart, width="stretch", key="official_draw_terminal_chart")
                st.dataframe(official_probabilities.head(24), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not fetch and simulate the official draw: {exc}")


if __name__ == "__main__":
    main()
