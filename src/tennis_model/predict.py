from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tennis_model.features import (
    build_feature_row,
    default_snapshot_row,
    pair_history_record,
    pair_history_to_states,
    snapshot_player_record,
)
from tennis_model.modeling import FEATURE_COLUMNS
from tennis_model.names import normalize_player_name


def load_snapshot(snapshot_path: str | Path) -> pd.DataFrame:
    snapshot = pd.read_csv(snapshot_path)
    snapshot["_lookup_name"] = snapshot["player_name"].map(normalize_player_name)
    return snapshot


def load_pair_history(pair_history_path: str | Path) -> pd.DataFrame:
    pair_history = pd.read_csv(pair_history_path)
    if "last_match_date" in pair_history.columns:
        pair_history["last_match_date"] = pd.to_datetime(pair_history["last_match_date"], errors="coerce")
    return pair_history


def pair_state_lookup(pair_history: pd.DataFrame | dict | None) -> dict:
    if pair_history is None:
        return {}
    if isinstance(pair_history, dict):
        return pair_history
    cached_states = pair_history.attrs.get("_pair_state_lookup")
    cached_length = pair_history.attrs.get("_pair_state_lookup_len")
    if cached_states is not None and cached_length == len(pair_history):
        return cached_states

    states = pair_history_to_states(pair_history)
    pair_history.attrs["_pair_state_lookup"] = states
    pair_history.attrs["_pair_state_lookup_len"] = len(pair_history)
    return states


def player_row(snapshot: pd.DataFrame, player_name: str) -> dict[str, Any]:
    lookup_name = normalize_player_name(player_name)
    matches = snapshot.loc[snapshot["_lookup_name"] == lookup_name]
    if matches.empty:
        return default_snapshot_row(player_name)
    return matches.iloc[0].to_dict()


def build_match_features(
    snapshot: pd.DataFrame,
    *,
    player_1: str,
    player_2: str,
    match_date: pd.Timestamp | None,
    surface: str,
    best_of: int,
    round_name: str,
    tourney_level: str,
    draw_size: int,
    pair_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    resolved_match_date = match_date
    if resolved_match_date is None:
        max_snapshot_date = pd.to_datetime(snapshot.get("last_match_date"), errors="coerce").max()
        resolved_match_date = max_snapshot_date if pd.notna(max_snapshot_date) else pd.Timestamp.today().normalize()

    player_1_record = snapshot_player_record(
        player_row(snapshot, player_1),
        surface,
        match_date=resolved_match_date,
        best_of=float(best_of),
    )
    player_2_record = snapshot_player_record(
        player_row(snapshot, player_2),
        surface,
        match_date=resolved_match_date,
        best_of=float(best_of),
    )
    pair_states = pair_state_lookup(pair_history)
    matchup = pair_history_record(
        pair_states,
        player_1_name=player_1,
        player_2_name=player_2,
        surface=surface,
        match_date=resolved_match_date,
    )
    row = build_feature_row(
        player_1_record,
        player_2_record,
        matchup=matchup,
        surface=surface,
        best_of=float(best_of),
        round_name=round_name,
        tourney_level=tourney_level,
        draw_size=float(draw_size),
    )
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_match_probability(
    model_path: str | Path,
    snapshot: pd.DataFrame,
    *,
    player_1: str,
    player_2: str,
    match_date: str | None = None,
    surface: str,
    best_of: int,
    round_name: str,
    tourney_level: str,
    draw_size: int,
    pair_history: pd.DataFrame | None = None,
) -> float:
    model = joblib.load(model_path)
    return predict_match_probability_with_model(
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


def predict_match_probability_with_model(
    model: Any,
    snapshot: pd.DataFrame,
    *,
    player_1: str,
    player_2: str,
    match_date: str | pd.Timestamp | None = None,
    surface: str,
    best_of: int,
    round_name: str,
    tourney_level: str,
    draw_size: int,
    pair_history: pd.DataFrame | None = None,
) -> float:
    resolved_match_date = (
        pd.Timestamp(match_date) if match_date is not None else None
    )
    features = build_match_features(
        snapshot,
        player_1=player_1,
        player_2=player_2,
        match_date=resolved_match_date,
        surface=surface,
        best_of=best_of,
        round_name=round_name,
        tourney_level=tourney_level,
        draw_size=draw_size,
        pair_history=pair_history,
    )
    return float(model.predict_proba(features)[:, 1][0])
