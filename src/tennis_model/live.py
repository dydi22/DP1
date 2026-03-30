from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_model.bracket import BYE_NAMES
from tennis_model.features import (
    DEFAULT_ELO_CONFIG,
    EloConfig,
    PlayerState,
    StatLine,
    build_match_stat_line,
    expected_score,
    pair_history_frame_from_states,
    pair_history_record,
    pair_history_to_states,
    player_key,
    player_state_to_live_row,
    snapshot_from_live_state,
    state_from_live_state_row,
    update_pair_history,
    update_elo,
    update_player_metadata,
    update_player_result,
)
from tennis_model.names import normalize_player_name
from tennis_model.predict import pair_state_lookup, predict_match_probability_with_model


RESULT_STAT_PREFIXES = (
    "ace",
    "df",
    "svpt",
    "1stIn",
    "1stWon",
    "2ndWon",
    "bpSaved",
    "bpFaced",
)


@dataclass
class BracketNode:
    round_name: str
    left: "BracketNode | str"
    right: "BracketNode | str"


def load_live_state(live_state_path: str | Path) -> pd.DataFrame:
    live_state = pd.read_csv(live_state_path)
    live_state["_lookup_name"] = live_state["player_name"].map(normalize_player_name)
    return live_state


def load_pair_history_state(pair_history_path: str | Path) -> pd.DataFrame:
    pair_history = pd.read_csv(pair_history_path)
    if "last_match_date" in pair_history.columns:
        pair_history["last_match_date"] = pd.to_datetime(pair_history["last_match_date"], errors="coerce")
    return pair_history


def _state_key_from_name(player_name: str) -> str:
    return player_key(None, player_name)


def live_state_to_states(live_state: pd.DataFrame) -> dict[str, PlayerState]:
    states: dict[str, PlayerState] = {}
    for row in live_state.to_dict("records"):
        player_name = str(row.get("player_name", "Unknown Player"))
        key = str(row.get("player_key") or _state_key_from_name(player_name))
        states[key] = state_from_live_state_row(row, player_name)
    return states


def states_to_live_state_frame(states: dict[str, PlayerState]) -> pd.DataFrame:
    rows = [
        player_state_to_live_row(player_key_value, state)
        for player_key_value, state in states.items()
    ]
    frame = pd.DataFrame(rows).sort_values("player_name").reset_index(drop=True)
    frame["_lookup_name"] = frame["player_name"].map(normalize_player_name)
    return frame


def _player_columns_for_prefix(prefix: str) -> set[str]:
    return {f"{prefix}_{field_name}" for field_name in RESULT_STAT_PREFIXES}


def _row_has_stats(row: pd.Series, prefix: str) -> bool:
    return any(column in row.index and pd.notna(row[column]) for column in _player_columns_for_prefix(prefix))


def _stat_line_from_result_row(row: pd.Series, prefix: str, opponent_prefix: str) -> StatLine:
    if not _row_has_stats(row, prefix):
        return StatLine()
    return build_match_stat_line(
        ace=row.get(f"{prefix}_ace"),
        double_fault=row.get(f"{prefix}_df"),
        serve_points_total=row.get(f"{prefix}_svpt"),
        first_serves_in=row.get(f"{prefix}_1stIn"),
        first_serve_points_won=row.get(f"{prefix}_1stWon"),
        second_serve_points_won=row.get(f"{prefix}_2ndWon"),
        break_points_saved=row.get(f"{prefix}_bpSaved"),
        break_points_faced=row.get(f"{prefix}_bpFaced"),
        opponent_serve_points_total=row.get(f"{opponent_prefix}_svpt"),
        opponent_first_serve_points_won=row.get(f"{opponent_prefix}_1stWon"),
        opponent_second_serve_points_won=row.get(f"{opponent_prefix}_2ndWon"),
    )


def _row_value(row: pd.Series, column: str, default: Any) -> Any:
    return row[column] if column in row.index and pd.notna(row[column]) else default


def apply_completed_matches(
    live_state: pd.DataFrame,
    pair_history: pd.DataFrame | dict | None,
    completed_results: pd.DataFrame,
    *,
    default_surface: str,
    default_best_of: int,
    default_tourney_level: str,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | dict]:
    states = live_state_to_states(live_state)
    pair_states = pair_history_to_states(pair_history)

    result_frame = completed_results.copy()
    if "match_date" in result_frame.columns:
        result_frame["match_date"] = pd.to_datetime(result_frame["match_date"], errors="coerce")
    else:
        result_frame["match_date"] = pd.NaT

    sort_columns = [column for column in ["match_date", "round_name", "player_1", "player_2"] if column in result_frame.columns]
    if sort_columns:
        result_frame = result_frame.sort_values(sort_columns, na_position="last").reset_index(drop=True)

    for row in result_frame.to_dict("records"):
        player_1_name = str(row["player_1"])
        player_2_name = str(row["player_2"])
        winner_name = str(row["winner"])
        loser_name = player_2_name if normalize_player_name(winner_name) == normalize_player_name(player_1_name) else player_1_name

        surface = str(row.get("surface") or default_surface)
        best_of = int(row.get("best_of") or default_best_of)
        tourney_level = str(row.get("tourney_level") or default_tourney_level)
        match_date = pd.Timestamp(row.get("match_date")) if pd.notna(row.get("match_date")) else pd.Timestamp.today().normalize()

        player_1_key = next(
            (key for key, state in states.items() if normalize_player_name(state.name) == normalize_player_name(player_1_name)),
            _state_key_from_name(player_1_name),
        )
        player_2_key = next(
            (key for key, state in states.items() if normalize_player_name(state.name) == normalize_player_name(player_2_name)),
            _state_key_from_name(player_2_name),
        )

        player_1_state = states.setdefault(player_1_key, PlayerState(name=player_1_name))
        player_2_state = states.setdefault(player_2_key, PlayerState(name=player_2_name))

        player_1_state.apply_decay(match_date, config=elo_config)
        player_2_state.apply_decay(match_date, config=elo_config)

        update_player_metadata(
            player_1_state,
            player_1_name,
            row.get("player_1_rank"),
            row.get("player_1_rank_points"),
            row.get("player_1_age"),
            row.get("player_1_height"),
            row.get("player_1_hand"),
        )
        update_player_metadata(
            player_2_state,
            player_2_name,
            row.get("player_2_rank"),
            row.get("player_2_rank_points"),
            row.get("player_2_age"),
            row.get("player_2_height"),
            row.get("player_2_hand"),
        )

        player_1_stat_line = _stat_line_from_result_row(pd.Series(row), "p1", "p2")
        player_2_stat_line = _stat_line_from_result_row(pd.Series(row), "p2", "p1")

        player_1_won = normalize_player_name(winner_name) == normalize_player_name(player_1_name)
        player_1_expected = expected_score(player_1_state.elo, player_2_state.elo)
        player_2_expected = 1.0 - player_1_expected
        update_elo(
            player_1_state,
            player_2_state,
            surface=surface,
            a_won=player_1_won,
            best_of=float(best_of),
            tourney_level=tourney_level,
            match_date=match_date,
            config=elo_config,
        )
        update_player_result(
            player_1_state,
            surface,
            won=player_1_won,
            match_date=match_date,
            best_of=float(best_of),
            stat_line=player_1_stat_line,
            match_minutes=row.get("minutes"),
            performance_residual=(1.0 if player_1_won else 0.0) - player_1_expected,
        )
        update_player_result(
            player_2_state,
            surface,
            won=not player_1_won,
            match_date=match_date,
            best_of=float(best_of),
            stat_line=player_2_stat_line,
            match_minutes=row.get("minutes"),
            performance_residual=(0.0 if player_1_won else 1.0) - player_2_expected,
        )

        winner_state = player_1_state if player_1_won else player_2_state
        winner_state.name = winner_name
        loser_state = player_2_state if player_1_won else player_1_state
        loser_state.name = loser_name
        update_pair_history(
            pair_states,
            player_1_name=player_1_name,
            player_2_name=player_2_name,
            winner_name=winner_name,
            surface=surface,
            match_date=match_date,
        )

    updated_live_state = states_to_live_state_frame(states)
    updated_snapshot = snapshot_from_live_state(updated_live_state.drop(columns=["_lookup_name"]))
    updated_pair_history = pair_states if isinstance(pair_history, dict) else pair_history_frame_from_states(pair_states)
    return updated_live_state.drop(columns=["_lookup_name"]), updated_snapshot, updated_pair_history


def load_draw_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_completed_results(results_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(results_path)
    required = {"player_1", "player_2", "winner", "round_name"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Completed results CSV is missing required columns: {sorted(missing)}")
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame


def audit_completed_predictions(
    model: Any,
    live_state: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    completed_results: pd.DataFrame,
    *,
    default_surface: str,
    default_best_of: int,
    default_tourney_level: str,
    draw_size: int,
    round_labels: list[str] | None = None,
) -> pd.DataFrame:
    audit_live_state = live_state.copy()
    audit_pair_history = pair_state_lookup(pair_history)
    results_frame = completed_results.copy()
    if "match_date" in results_frame.columns:
        results_frame["match_date"] = pd.to_datetime(results_frame["match_date"], errors="coerce")

    if round_labels is not None:
        round_order = {label: index for index, label in enumerate(round_labels)}
        results_frame["round_order"] = results_frame["round_name"].map(round_order).fillna(len(round_order))
        sort_columns = ["match_date", "round_order", "player_1", "player_2"]
    else:
        sort_columns = ["match_date", "round_name", "player_1", "player_2"]
    results_frame = results_frame.sort_values(sort_columns, na_position="last").reset_index(drop=True)
    if "round_order" in results_frame.columns:
        results_frame = results_frame.drop(columns=["round_order"])

    rows: list[dict[str, Any]] = []
    for row in results_frame.to_dict("records"):
        runtime_snapshot = snapshot_from_live_state(audit_live_state.drop(columns=["_lookup_name"]))
        runtime_snapshot["_lookup_name"] = runtime_snapshot["player_name"].map(normalize_player_name)

        surface = str(row.get("surface") or default_surface)
        best_of = int(row.get("best_of") or default_best_of)
        tourney_level = str(row.get("tourney_level") or default_tourney_level)
        match_date = row.get("match_date")
        round_name = str(row["round_name"])
        player_1 = str(row["player_1"])
        player_2 = str(row["player_2"])
        winner = str(row["winner"])
        player_1_win_probability = predict_match_probability_with_model(
            model,
            runtime_snapshot,
            player_1=player_1,
            player_2=player_2,
            match_date=match_date,
            surface=surface,
            best_of=best_of,
            round_name=round_name,
            tourney_level=tourney_level,
            draw_size=draw_size,
            pair_history=audit_pair_history or None,
        )

        actual_player_1_win = normalize_player_name(winner) == normalize_player_name(player_1)
        favorite = player_1 if player_1_win_probability >= 0.5 else player_2
        correct_pick = (
            (favorite == player_1 and actual_player_1_win)
            or (favorite == player_2 and not actual_player_1_win)
        )
        rows.append(
            {
                "match_date": pd.Timestamp(match_date).date().isoformat() if pd.notna(match_date) else None,
                "round_name": round_name,
                "player_1": player_1,
                "player_2": player_2,
                "winner": winner,
                "match_id": row.get("match_id"),
                "favorite": favorite,
                "player_1_win_probability": player_1_win_probability,
                "favorite_win_probability": player_1_win_probability if favorite == player_1 else 1 - player_1_win_probability,
                "actual_player_1_win": int(actual_player_1_win),
                "correct_pick": bool(correct_pick),
                "brier_error": float((player_1_win_probability - int(actual_player_1_win)) ** 2),
                "log_loss": float(
                    -np.log(max(player_1_win_probability, 1e-12))
                    if actual_player_1_win
                    else -np.log(max(1 - player_1_win_probability, 1e-12))
                ),
                "confidence": abs(player_1_win_probability - 0.5) * 200,
            }
        )

        single_match_frame = pd.DataFrame([row])
        audit_live_state, _, audit_pair_history = apply_completed_matches(
            audit_live_state,
            audit_pair_history,
            single_match_frame,
            default_surface=default_surface,
            default_best_of=default_best_of,
            default_tourney_level=default_tourney_level,
        )
        audit_live_state["_lookup_name"] = audit_live_state["player_name"].map(normalize_player_name)

    return pd.DataFrame(rows)


def _result_lookup(frame: pd.DataFrame) -> dict[tuple[str, frozenset[str]], str]:
    lookup: dict[tuple[str, frozenset[str]], str] = {}
    for row in frame.to_dict("records"):
        round_name = str(row["round_name"])
        players = frozenset(
            {
                normalize_player_name(str(row["player_1"])),
                normalize_player_name(str(row["player_2"])),
            }
        )
        lookup[(round_name, players)] = str(row["winner"])
    return lookup


def _build_bracket_forest(
    *,
    first_round_pairs: list[tuple[str, str]],
    round_labels: list[str],
) -> list[BracketNode]:
    current_level: list[BracketNode] = [
        BracketNode(round_name=round_labels[0], left=player_1, right=player_2)
        for player_1, player_2 in first_round_pairs
    ]
    for round_label in round_labels[1:]:
        if len(current_level) % 2 != 0:
            raise ValueError("Current level does not pair evenly into the next round.")
        current_level = [
            BracketNode(round_name=round_label, left=current_level[index], right=current_level[index + 1])
            for index in range(0, len(current_level), 2)
        ]
    return current_level


def _child_entrant_if_known(
    child: BracketNode | str,
    *,
    lookup: dict[tuple[str, frozenset[str]], str],
) -> str | None:
    if isinstance(child, str):
        return child
    return _node_winner_if_known(child, lookup=lookup)


def _node_winner_if_known(
    node: BracketNode,
    *,
    lookup: dict[tuple[str, frozenset[str]], str],
) -> str | None:
    left_entrant = _child_entrant_if_known(node.left, lookup=lookup)
    right_entrant = _child_entrant_if_known(node.right, lookup=lookup)

    if left_entrant is None or right_entrant is None:
        return None
    if left_entrant in BYE_NAMES and right_entrant in BYE_NAMES:
        return None
    if left_entrant in BYE_NAMES:
        return right_entrant
    if right_entrant in BYE_NAMES:
        return left_entrant

    winner = lookup.get(
        (
            node.round_name,
            frozenset({normalize_player_name(left_entrant), normalize_player_name(right_entrant)}),
        )
    )
    if winner is None:
        return None
    normalized_winner = normalize_player_name(winner)
    if normalized_winner == normalize_player_name(left_entrant):
        return left_entrant
    if normalized_winner == normalize_player_name(right_entrant):
        return right_entrant
    return winner


def _collect_pending_matches(
    node: BracketNode,
    *,
    lookup: dict[tuple[str, frozenset[str]], str],
    rows: list[dict[str, Any]],
) -> None:
    winner = _node_winner_if_known(node, lookup=lookup)
    if winner is not None:
        return

    left_entrant = _child_entrant_if_known(node.left, lookup=lookup)
    right_entrant = _child_entrant_if_known(node.right, lookup=lookup)
    if left_entrant is not None and right_entrant is not None:
        if left_entrant in BYE_NAMES and right_entrant in BYE_NAMES:
            return
        if left_entrant in BYE_NAMES or right_entrant in BYE_NAMES:
            return
        rows.append(
            {
                "player_1": left_entrant,
                "player_2": right_entrant,
                "round_name": node.round_name,
            }
        )
        return

    if isinstance(node.left, BracketNode):
        _collect_pending_matches(node.left, lookup=lookup, rows=rows)
    if isinstance(node.right, BracketNode):
        _collect_pending_matches(node.right, lookup=lookup, rows=rows)


def build_pending_matches(
    *,
    first_round_pairs: list[tuple[str, str]],
    round_labels: list[str],
    completed_results: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    lookup = _result_lookup(completed_results)
    forest = _build_bracket_forest(
        first_round_pairs=first_round_pairs,
        round_labels=round_labels,
    )
    rows: list[dict[str, Any]] = []
    for root in forest:
        _collect_pending_matches(root, lookup=lookup, rows=rows)

    pending_matches = pd.DataFrame(rows)
    if pending_matches.empty:
        return pending_matches, [], ""

    round_order = {label: index for index, label in enumerate(round_labels)}
    pending_matches["round_order"] = pending_matches["round_name"].map(round_order).fillna(len(round_order))
    pending_matches = pending_matches.sort_values(
        ["round_order", "player_1", "player_2"]
    ).drop(columns=["round_order"]).reset_index(drop=True)
    active_rounds = list(dict.fromkeys(pending_matches["round_name"].tolist()))
    return pending_matches, active_rounds, active_rounds[0]


def score_current_round(
    model: Any,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    pending_matches: pd.DataFrame,
    *,
    match_date: str | None,
    surface: str,
    best_of: int,
    tourney_level: str,
    draw_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pending_matches.to_dict("records"):
        player_1 = str(row["player_1"])
        player_2 = str(row["player_2"])
        round_name = str(row["round_name"])
        if player_1 in BYE_NAMES and player_2 in BYE_NAMES:
            continue
        if player_1 in BYE_NAMES:
            probability = 0.0
            favorite = player_2
        elif player_2 in BYE_NAMES:
            probability = 1.0
            favorite = player_1
        else:
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
        rows.append(
            {
                "player_1": player_1,
                "player_2": player_2,
                "round_name": round_name,
                "favorite": favorite,
                "player_1_win_probability": probability,
                "confidence": abs(probability - 0.5) * 200,
            }
        )
    return pd.DataFrame(rows)


def simulate_remaining_tournament(
    *,
    model_path: str | Path,
    snapshot: pd.DataFrame,
    pair_history: pd.DataFrame | None,
    first_round_pairs: list[tuple[str, str]],
    completed_results: pd.DataFrame,
    round_labels: list[str],
    terminal_label: str,
    tournament_date: str | None,
    surface: str,
    best_of: int,
    tourney_level: str,
    simulations: int,
) -> pd.DataFrame:
    if not first_round_pairs:
        return pd.DataFrame(columns=["player_name", terminal_label])
    lookup = _result_lookup(completed_results)
    forest = _build_bracket_forest(
        first_round_pairs=first_round_pairs,
        round_labels=round_labels,
    )

    entrants = sorted(
        {
            player
            for pair in first_round_pairs
            for player in pair
            if player not in BYE_NAMES
        }
    )
    if not entrants:
        return pd.DataFrame(columns=["player_name", terminal_label])

    advancement_labels = round_labels[1:] + [terminal_label]
    counts: dict[str, dict[str, int]] = {
        player: {label: 0 for label in advancement_labels}
        for player in entrants
    }

    model = joblib.load(model_path)
    rng = np.random.default_rng(42)
    probability_cache: dict[tuple[str, str, str | None, str, int, str, str, int], float] = {}
    draw_size = len(first_round_pairs) * 2

    def resolve_node(node: BracketNode, next_label: str) -> str | None:
        known_winner = _node_winner_if_known(node, lookup=lookup)
        if known_winner is not None:
            if known_winner not in BYE_NAMES:
                counts[known_winner][next_label] += 1
            return known_winner

        if isinstance(node.left, BracketNode):
            left_entrant = resolve_node(node.left, node.round_name)
        else:
            left_entrant = node.left
        if isinstance(node.right, BracketNode):
            right_entrant = resolve_node(node.right, node.round_name)
        else:
            right_entrant = node.right

        if left_entrant is None or right_entrant is None:
            return None
        if left_entrant in BYE_NAMES and right_entrant in BYE_NAMES:
            return None
        if left_entrant in BYE_NAMES:
            winner = right_entrant
        elif right_entrant in BYE_NAMES:
            winner = left_entrant
        else:
            cache_key = (
                left_entrant,
                right_entrant,
                tournament_date,
                surface,
                best_of,
                node.round_name,
                tourney_level,
                draw_size,
            )
            probability = probability_cache.get(cache_key)
            if probability is None:
                probability = predict_match_probability_with_model(
                    model,
                    snapshot,
                    player_1=left_entrant,
                    player_2=right_entrant,
                    match_date=tournament_date,
                    surface=surface,
                    best_of=best_of,
                    round_name=node.round_name,
                    tourney_level=tourney_level,
                    draw_size=draw_size,
                    pair_history=pair_history,
                )
                probability_cache[cache_key] = probability
            winner = left_entrant if rng.random() < probability else right_entrant

        if winner not in BYE_NAMES:
            counts[winner][next_label] += 1
        return winner

    for _ in range(simulations):
        for root in forest:
            resolve_node(root, terminal_label)

    rows: list[dict[str, Any]] = []
    for player in entrants:
        row = {"player_name": player}
        for label in advancement_labels:
            row[label] = counts[player][label] / simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values(terminal_label, ascending=False).reset_index(drop=True)
