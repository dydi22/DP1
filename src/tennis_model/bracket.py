from __future__ import annotations

from collections import Counter
from math import log2
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_model.predict import predict_match_probability_with_model


ROUND_LABELS = {
    2: "F",
    4: "SF",
    8: "QF",
    16: "R16",
    32: "R32",
    64: "R64",
    128: "R128",
}

BYE_NAMES = {"BYE", "Bye", "bye"}


def round_sequence(field_size: int) -> list[str]:
    if field_size < 2 or field_size & (field_size - 1):
        raise ValueError("Field size must be a power of 2.")

    rounds = int(log2(field_size))
    labels = []
    current_size = field_size
    for _ in range(rounds):
        labels.append(ROUND_LABELS.get(current_size, f"R{current_size}"))
        current_size //= 2
    return labels


def load_bracket(bracket_csv: str | Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(bracket_csv)
    required = {"player_1", "player_2"}
    if not required.issubset(frame.columns):
        raise ValueError("Bracket CSV must include player_1 and player_2 columns.")
    return list(frame[["player_1", "player_2"]].itertuples(index=False, name=None))


def simulate_tournament(
    *,
    model_path: str | Path,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    first_round_pairs: list[tuple[str, str]],
    tournament_date: str | None,
    surface: str,
    best_of: int,
    tourney_level: str,
    draw_size: int,
    simulations: int,
    round_labels: list[str] | None = None,
    terminal_label: str = "Champion",
) -> pd.DataFrame:
    raw_entrants = [player for pair in first_round_pairs for player in pair]
    if len(raw_entrants) != draw_size:
        raise ValueError(
            f"Bracket has {len(raw_entrants)} slots but draw_size is {draw_size}."
        )
    entrants = [player for player in raw_entrants if player not in BYE_NAMES]

    rounds = round_labels or round_sequence(draw_size)
    advancement_labels = rounds[1:] + [terminal_label]
    model = joblib.load(model_path)
    rng = np.random.default_rng(42)

    counts: dict[str, Counter[str]] = {player: Counter() for player in entrants}
    probability_cache: dict[tuple[str, str, str | None, str, int, str, str, int], float] = {}
    for player in entrants:
        counts[player][rounds[0]] += simulations

    for _ in range(simulations):
        current_pairs = first_round_pairs
        for round_name, next_label in zip(rounds, advancement_labels):
            winners = []
            for player_1, player_2 in current_pairs:
                if player_1 in BYE_NAMES and player_2 in BYE_NAMES:
                    continue
                if player_1 in BYE_NAMES:
                    winners.append(player_2)
                    counts[player_2][next_label] += 1
                    continue
                if player_2 in BYE_NAMES:
                    winners.append(player_1)
                    counts[player_1][next_label] += 1
                    continue
                cache_key = (
                    player_1,
                    player_2,
                    tournament_date,
                    surface,
                    best_of,
                    round_name,
                    tourney_level,
                    draw_size,
                )
                probability = probability_cache.get(cache_key)
                if probability is None:
                    probability = predict_match_probability_with_model(
                        model,
                        snapshot,
                        player_1=player_1,
                        player_2=player_2,
                        match_date=tournament_date,
                        surface=surface,
                        best_of=best_of,
                        round_name=round_name,
                        tourney_level=tourney_level,
                        draw_size=draw_size,
                        pair_history=pair_history,
                    )
                    probability_cache[cache_key] = probability
                winner = player_1 if rng.random() < probability else player_2
                winners.append(winner)
                counts[winner][next_label] += 1

            current_pairs = []
            if len(winners) > 1:
                current_pairs = [
                    (winners[index], winners[index + 1])
                    for index in range(0, len(winners), 2)
                ]

    rows: list[dict[str, Any]] = []
    ordered_labels = rounds + [terminal_label]
    for player, counter in counts.items():
        row = {"player_name": player}
        for label in ordered_labels:
            row[label] = counter[label] / simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values(terminal_label, ascending=False).reset_index(drop=True)
