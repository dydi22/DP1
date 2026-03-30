from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_model.names import normalize_player_name


def _history_key(row: dict[str, Any]) -> str:
    match_date = str(row.get("match_date") or "").strip()
    round_name = str(row.get("round_name") or "").strip()
    players = sorted(
        [
            normalize_player_name(str(row.get("player_1", ""))),
            normalize_player_name(str(row.get("player_2", ""))),
        ]
    )
    if match_date or round_name or any(players):
        return "|".join([match_date, round_name, players[0], players[1]])
    match_id = row.get("match_id")
    if pd.notna(match_id) and str(match_id).strip():
        return str(match_id).strip().upper()
    return "|".join(
        [
            match_date,
            round_name,
            players[0],
            players[1],
        ]
    )


def history_bucket_frame(history: pd.DataFrame, *, bucket_count: int = 10) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=["bucket", "matches", "avg_predicted_probability", "actual_win_rate", "accuracy"]
        )
    frame = history.copy()
    frame["bucket"] = pd.cut(
        frame["player_1_win_probability"],
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
                "avg_predicted_probability": float(bucket_frame["player_1_win_probability"].mean()),
                "actual_win_rate": float(bucket_frame["actual_player_1_win"].mean()),
                "accuracy": float(bucket_frame["correct_pick"].mean()),
            }
        )
    return pd.DataFrame(rows)


def upsert_prediction_history(
    history_dir: str | Path,
    audit_frame: pd.DataFrame,
    *,
    source_name: str,
) -> tuple[Path, Path, Path]:
    output_dir = Path(history_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "prediction_history.csv"
    bucket_path = output_dir / "prediction_history_buckets.csv"
    summary_path = output_dir / "prediction_history_summary.json"

    history_columns = [
        "history_key",
        "prediction_source",
        "match_id",
        "match_date",
        "round_name",
        "player_1",
        "player_2",
        "winner",
        "favorite",
        "player_1_win_probability",
        "favorite_win_probability",
        "actual_player_1_win",
        "correct_pick",
        "brier_error",
        "log_loss",
        "confidence",
    ]
    frame = audit_frame.copy()
    if frame.empty:
        if not history_path.exists():
            pd.DataFrame(columns=history_columns).to_csv(history_path, index=False)
        history = (
            pd.read_csv(history_path)
            if history_path.exists() and history_path.stat().st_size > 1
            else pd.DataFrame(columns=history_columns)
        )
    else:
        frame["prediction_source"] = source_name
        frame["history_key"] = [_history_key(row) for row in frame.to_dict("records")]
        existing = (
            pd.read_csv(history_path)
            if history_path.exists() and history_path.stat().st_size > 1
            else pd.DataFrame(columns=history_columns)
        )
        if not existing.empty:
            existing["history_key"] = [_history_key(row) for row in existing.to_dict("records")]
        if existing.empty:
            history = frame.copy()
        else:
            history = pd.concat([existing, frame], ignore_index=True)
        history = history.drop_duplicates(subset=["history_key"], keep="last").sort_values(
            ["match_date", "round_name", "player_1", "player_2"],
            na_position="last",
        ).reset_index(drop=True)
        history.to_csv(history_path, index=False)

    if history.empty:
        bucket_frame = pd.DataFrame(
            columns=["bucket", "matches", "avg_predicted_probability", "actual_win_rate", "accuracy"]
        )
        summary = {"matches": 0, "accuracy": None, "avg_log_loss": None, "avg_brier_error": None}
    else:
        bucket_frame = history_bucket_frame(history)
        summary = {
            "matches": int(len(history)),
            "accuracy": float(history["correct_pick"].mean()),
            "avg_log_loss": float(history["log_loss"].mean()),
            "avg_brier_error": float(history["brier_error"].mean()),
        }
    bucket_frame.to_csv(bucket_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return history_path, bucket_path, summary_path
