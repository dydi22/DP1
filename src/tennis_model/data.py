from __future__ import annotations

from pathlib import Path

import pandas as pd


MATCH_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "indoor",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "best_of",
    "round",
    "minutes",
    "winner_id",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]


def load_matches(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("atp_matches_[0-9][0-9][0-9][0-9].csv"))
    files.extend(sorted(data_path.glob("atp_matches_[0-9][0-9][0-9][0-9]_ongoing.csv")))
    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(
            f"No ATP match files were found in {data_path}. "
            "Expected files like atp_matches_2024.csv."
        )

    frames = []
    for file_path in files:
        frame = pd.read_csv(
            file_path,
            usecols=lambda column: column in MATCH_COLUMNS,
            low_memory=False,
        )
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    matches = matches.dropna(
        subset=[
            "tourney_date",
            "winner_name",
            "loser_name",
            "surface",
            "best_of",
        ]
    ).copy()

    matches["match_num"] = pd.to_numeric(matches["match_num"], errors="coerce").fillna(0)
    matches["draw_size"] = pd.to_numeric(matches["draw_size"], errors="coerce")
    matches["best_of"] = pd.to_numeric(matches["best_of"], errors="coerce")
    matches["match_num"] = pd.to_numeric(matches["match_num"], errors="coerce").fillna(0)

    dedupe_columns = [
        "tourney_id",
        "tourney_date",
        "match_num",
        "round",
        "winner_name",
        "loser_name",
    ]
    matches = matches.drop_duplicates(subset=dedupe_columns, keep="last")

    matches = matches.sort_values(
        by=["tourney_date", "tourney_id", "match_num", "winner_name", "loser_name"]
    ).reset_index(drop=True)
    return matches
