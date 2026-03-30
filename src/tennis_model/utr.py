from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tennis_model.names import normalize_player_name


REQUIRED_UTR_COLUMNS = {"player_name", "rating_date", "utr_singles"}
REQUIRED_ALIAS_COLUMNS = {"source_name", "canonical_name"}


def load_name_aliases(alias_csv: str | Path | None) -> dict[str, str]:
    if alias_csv is None:
        return {}

    alias_frame = pd.read_csv(alias_csv)
    missing = REQUIRED_ALIAS_COLUMNS - set(alias_frame.columns)
    if missing:
        raise ValueError(
            f"Alias CSV is missing required columns: {sorted(missing)}"
        )

    alias_map: dict[str, str] = {}
    for row in alias_frame.itertuples(index=False):
        alias_map[normalize_player_name(row.source_name)] = str(row.canonical_name)
    return alias_map


def load_utr_history(
    utr_history_csv: str | Path,
    *,
    alias_csv: str | Path | None = None,
) -> pd.DataFrame:
    history = pd.read_csv(utr_history_csv)
    missing = REQUIRED_UTR_COLUMNS - set(history.columns)
    if missing:
        raise ValueError(
            f"UTR history CSV is missing required columns: {sorted(missing)}"
        )

    aliases = load_name_aliases(alias_csv)
    history = history.copy()
    history["rating_date"] = pd.to_datetime(history["rating_date"], errors="coerce")
    history["utr_singles"] = pd.to_numeric(history["utr_singles"], errors="coerce")
    history = history.dropna(subset=["player_name", "rating_date", "utr_singles"])
    history["aligned_player_name"] = history["player_name"].map(str)
    history["aligned_player_name"] = history["aligned_player_name"].map(
        lambda name: aliases.get(normalize_player_name(name), name)
    )
    history["player_lookup"] = history["aligned_player_name"].map(normalize_player_name)
    history = history.sort_values(["player_lookup", "rating_date"]).reset_index(drop=True)
    return history


@dataclass
class UTRTracker:
    events_by_player: dict[str, list[tuple[pd.Timestamp, float]]]
    latest_by_player: dict[str, float]
    positions: dict[str, int] = field(default_factory=dict)
    current_by_player: dict[str, float | None] = field(default_factory=dict)

    @classmethod
    def from_history(cls, history: pd.DataFrame) -> "UTRTracker":
        events_by_player: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        latest_by_player: dict[str, float] = {}

        for row in history.itertuples(index=False):
            events = events_by_player.setdefault(row.player_lookup, [])
            events.append((row.rating_date, float(row.utr_singles)))
            latest_by_player[row.player_lookup] = float(row.utr_singles)

        return cls(events_by_player=events_by_player, latest_by_player=latest_by_player)

    def rating_for_player_on_date(
        self,
        player_name: str,
        as_of_date: pd.Timestamp,
    ) -> float | None:
        lookup = normalize_player_name(player_name)
        events = self.events_by_player.get(lookup, [])
        if not events:
            return None

        position = self.positions.get(lookup, 0)
        current_rating = self.current_by_player.get(lookup)
        while position < len(events) and events[position][0] <= as_of_date:
            current_rating = events[position][1]
            position += 1

        self.positions[lookup] = position
        self.current_by_player[lookup] = current_rating
        return current_rating

    def latest_rating(self, player_name: str) -> float | None:
        return self.latest_by_player.get(normalize_player_name(player_name))
