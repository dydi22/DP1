from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tennis_model.names import normalize_player_name


SURFACES = ("Hard", "Clay", "Grass", "Carpet")
STAT_LINE_FIELDS = (
    "serve_points_total",
    "serve_points_won",
    "first_serves_in",
    "first_serve_points_won",
    "second_serve_points_total",
    "second_serve_points_won",
    "aces",
    "double_faults",
    "break_points_saved",
    "break_points_faced",
    "return_points_total",
    "return_points_won",
)
BASE_ELO = 1500.0
RECENT_RESULTS_WINDOW = 10
FATIGUE_WINDOW_DAYS = 30
OVERALL_ELO_HALF_LIFE_DAYS = 240.0
RECENT_ELO_HALF_LIFE_DAYS = 120.0
BEST_OF_FIVE_ELO_HALF_LIFE_DAYS = 365.0
H2H_RECENT_WINDOW = 5
K_INACTIVITY_CAP_DAYS = 180

TOURNEY_LEVEL_K_MULTIPLIER = {
    "G": 1.08,
    "M": 1.04,
    "A": 1.00,
    "D": 0.98,
    "F": 0.95,
    "O": 1.00,
    "C": 0.95,
    "S": 0.92,
}


@dataclass(frozen=True)
class EloConfig:
    base_k: float = 28.0
    overall_half_life_days: float = OVERALL_ELO_HALF_LIFE_DAYS
    recent_half_life_days: float = RECENT_ELO_HALF_LIFE_DAYS
    best_of_five_half_life_days: float = BEST_OF_FIVE_ELO_HALF_LIFE_DAYS
    surface_half_life_days: float = OVERALL_ELO_HALF_LIFE_DAYS
    experience_min_multiplier: float = 0.8
    experience_max_multiplier: float = 1.35
    experience_match_cap: int = 110
    experience_divisor: float = 200.0
    inactivity_start_days: int = 14
    inactivity_cap_days: int = K_INACTIVITY_CAP_DAYS
    inactivity_max_multiplier: float = 1.35
    inactivity_divisor: float = 450.0
    recent_k_multiplier: float = 1.25
    surface_k_multiplier: float = 1.05
    best_of_five_k_multiplier: float = 1.10
    best_of_five_match_multiplier: float = 1.03


DEFAULT_ELO_CONFIG = EloConfig()


def safe_value(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def smooth_win_rate(
    wins: int,
    matches: int,
    prior_rate: float = 0.5,
    weight: int = 4,
) -> float:
    return (wins + prior_rate * weight) / (matches + weight)


def smooth_ratio(
    numerator: float,
    denominator: float,
    *,
    prior_rate: float,
    weight: float,
) -> float:
    if denominator < 0:
        denominator = 0.0
    return (numerator + prior_rate * weight) / (denominator + weight)


def player_key(player_id: Any, player_name: str) -> str:
    if pd.notna(player_id):
        player_id_str = str(player_id).strip()
        if player_id_str and player_id_str.lower() != "nan":
            if player_id_str.endswith(".0"):
                player_id_str = player_id_str[:-2]
            return f"id::{player_id_str}"
    return f"name::{normalize_player_name(player_name)}"


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def regress_toward_mean(rating: float, days_inactive: int, half_life_days: float) -> float:
    if days_inactive <= 0:
        return rating
    decay_factor = 0.5 ** (days_inactive / half_life_days)
    return BASE_ELO + (rating - BASE_ELO) * decay_factor


def experience_k_multiplier(matches_played: int, config: EloConfig = DEFAULT_ELO_CONFIG) -> float:
    return max(
        config.experience_min_multiplier,
        config.experience_max_multiplier - min(matches_played, config.experience_match_cap) / config.experience_divisor,
    )


def inactivity_k_multiplier(days_inactive: int, config: EloConfig = DEFAULT_ELO_CONFIG) -> float:
    if days_inactive <= config.inactivity_start_days:
        return 1.0
    clipped_days = min(days_inactive, config.inactivity_cap_days)
    return min(
        config.inactivity_max_multiplier,
        1.0 + (clipped_days - config.inactivity_start_days) / config.inactivity_divisor,
    )


def tournament_k_multiplier(tourney_level: str) -> float:
    return TOURNEY_LEVEL_K_MULTIPLIER.get(str(tourney_level), 1.0)


@dataclass
class StatLine:
    serve_points_total: float = 0.0
    serve_points_won: float = 0.0
    first_serves_in: float = 0.0
    first_serve_points_won: float = 0.0
    second_serve_points_total: float = 0.0
    second_serve_points_won: float = 0.0
    aces: float = 0.0
    double_faults: float = 0.0
    break_points_saved: float = 0.0
    break_points_faced: float = 0.0
    return_points_total: float = 0.0
    return_points_won: float = 0.0

    def add(self, other: "StatLine") -> None:
        self.serve_points_total += other.serve_points_total
        self.serve_points_won += other.serve_points_won
        self.first_serves_in += other.first_serves_in
        self.first_serve_points_won += other.first_serve_points_won
        self.second_serve_points_total += other.second_serve_points_total
        self.second_serve_points_won += other.second_serve_points_won
        self.aces += other.aces
        self.double_faults += other.double_faults
        self.break_points_saved += other.break_points_saved
        self.break_points_faced += other.break_points_faced
        self.return_points_total += other.return_points_total
        self.return_points_won += other.return_points_won


def combine_stat_lines(stat_lines: deque[StatLine]) -> StatLine:
    combined = StatLine()
    for stat_line in stat_lines:
        combined.add(stat_line)
    return combined


def stat_profile(stat_line: StatLine) -> dict[str, float]:
    return {
        "serve_win_rate": smooth_ratio(
            stat_line.serve_points_won,
            stat_line.serve_points_total,
            prior_rate=0.62,
            weight=25.0,
        ),
        "return_win_rate": smooth_ratio(
            stat_line.return_points_won,
            stat_line.return_points_total,
            prior_rate=0.38,
            weight=25.0,
        ),
        "first_serve_in_rate": smooth_ratio(
            stat_line.first_serves_in,
            stat_line.serve_points_total,
            prior_rate=0.60,
            weight=25.0,
        ),
        "first_serve_win_rate": smooth_ratio(
            stat_line.first_serve_points_won,
            stat_line.first_serves_in,
            prior_rate=0.70,
            weight=20.0,
        ),
        "second_serve_win_rate": smooth_ratio(
            stat_line.second_serve_points_won,
            stat_line.second_serve_points_total,
            prior_rate=0.50,
            weight=20.0,
        ),
        "ace_rate": smooth_ratio(
            stat_line.aces,
            stat_line.serve_points_total,
            prior_rate=0.05,
            weight=40.0,
        ),
        "double_fault_rate": smooth_ratio(
            stat_line.double_faults,
            stat_line.serve_points_total,
            prior_rate=0.03,
            weight=40.0,
        ),
        "bp_save_rate": smooth_ratio(
            stat_line.break_points_saved,
            stat_line.break_points_faced,
            prior_rate=0.60,
            weight=12.0,
        ),
    }


def numeric_or_zero(value: Any) -> float:
    numeric_value = safe_value(value)
    return 0.0 if numeric_value is None else numeric_value


def serialize_float_deque(values: deque[float]) -> str:
    return ";".join(f"{value:.6f}" for value in values)


def parse_float_deque(value: Any) -> deque[float]:
    parsed: deque[float] = deque(maxlen=RECENT_RESULTS_WINDOW)
    if pd.isna(value) or value is None or value == "":
        return parsed
    for item in str(value).split(";"):
        item = item.strip()
        if item:
            parsed.append(float(item))
    return parsed


def build_match_stat_line(
    *,
    ace: Any,
    double_fault: Any,
    serve_points_total: Any,
    first_serves_in: Any,
    first_serve_points_won: Any,
    second_serve_points_won: Any,
    break_points_saved: Any,
    break_points_faced: Any,
    opponent_serve_points_total: Any,
    opponent_first_serve_points_won: Any,
    opponent_second_serve_points_won: Any,
) -> StatLine:
    svpt = numeric_or_zero(serve_points_total)
    first_in = numeric_or_zero(first_serves_in)
    first_won = numeric_or_zero(first_serve_points_won)
    second_won = numeric_or_zero(second_serve_points_won)
    opponent_svpt = numeric_or_zero(opponent_serve_points_total)
    opponent_first_won = numeric_or_zero(opponent_first_serve_points_won)
    opponent_second_won = numeric_or_zero(opponent_second_serve_points_won)
    second_total = max(0.0, svpt - first_in)
    return_won = max(0.0, opponent_svpt - opponent_first_won - opponent_second_won)

    return StatLine(
        serve_points_total=svpt,
        serve_points_won=first_won + second_won,
        first_serves_in=first_in,
        first_serve_points_won=first_won,
        second_serve_points_total=second_total,
        second_serve_points_won=second_won,
        aces=numeric_or_zero(ace),
        double_faults=numeric_or_zero(double_fault),
        break_points_saved=numeric_or_zero(break_points_saved),
        break_points_faced=numeric_or_zero(break_points_faced),
        return_points_total=opponent_svpt,
        return_points_won=return_won,
    )


@dataclass
class PlayerState:
    name: str = ""
    matches: int = 0
    wins: int = 0
    elo: float = BASE_ELO
    recent_elo: float = BASE_ELO
    best_of_five_elo: float = BASE_ELO
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_RESULTS_WINDOW))
    recent_match_dates: deque[pd.Timestamp] = field(default_factory=deque)
    recent_performance_residuals: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_RESULTS_WINDOW))
    recent_match_minutes: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_RESULTS_WINDOW))
    stats: StatLine = field(default_factory=StatLine)
    surface_stats: dict[str, StatLine] = field(default_factory=lambda: defaultdict(StatLine))
    recent_stat_lines: deque[StatLine] = field(default_factory=lambda: deque(maxlen=RECENT_RESULTS_WINDOW))
    best_of_five_matches: int = 0
    best_of_five_wins: int = 0
    last_match_date: pd.Timestamp | None = None
    last_rank: float | None = None
    last_rank_points: float | None = None
    last_age: float | None = None
    last_height: float | None = None
    last_hand: str | None = None
    last_utr: float | None = None

    def overall_win_rate(self) -> float:
        return smooth_win_rate(self.wins, self.matches)

    def surface_win_rate(self, surface: str) -> float:
        return smooth_win_rate(self.surface_wins[surface], self.surface_matches[surface])

    def recent_form(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    def best_of_five_win_rate(self) -> float:
        return smooth_win_rate(self.best_of_five_wins, self.best_of_five_matches)

    def recent_adjusted_form(self) -> float:
        if not self.recent_performance_residuals:
            return 0.0
        return sum(self.recent_performance_residuals) / len(self.recent_performance_residuals)

    def recent_minutes_total(self) -> float:
        return sum(self.recent_match_minutes)

    def recent_minutes_average(self) -> float:
        if not self.recent_match_minutes:
            return 0.0
        return self.recent_minutes_total() / len(self.recent_match_minutes)

    def stat_profile(self) -> dict[str, float]:
        return stat_profile(self.stats)

    def surface_stat_profile(self, surface: str) -> dict[str, float]:
        return stat_profile(self.surface_stats[surface])

    def recent_stat_profile(self) -> dict[str, float]:
        return stat_profile(combine_stat_lines(self.recent_stat_lines))

    def matches_in_recent_window(self, as_of_date: pd.Timestamp) -> int:
        while self.recent_match_dates and (as_of_date - self.recent_match_dates[0]).days > FATIGUE_WINDOW_DAYS:
            self.recent_match_dates.popleft()
        return len(self.recent_match_dates)

    def apply_decay(self, as_of_date: pd.Timestamp, config: EloConfig = DEFAULT_ELO_CONFIG) -> None:
        if self.last_match_date is None:
            return

        days_inactive = max(0, (as_of_date - self.last_match_date).days)
        self.elo = regress_toward_mean(self.elo, days_inactive, config.overall_half_life_days)
        self.recent_elo = regress_toward_mean(
            self.recent_elo,
            days_inactive,
            config.recent_half_life_days,
        )
        self.best_of_five_elo = regress_toward_mean(
            self.best_of_five_elo,
            days_inactive,
            config.best_of_five_half_life_days,
        )
        for surface in SURFACES:
            self.surface_elo[surface] = regress_toward_mean(
                self.surface_elo[surface],
                days_inactive,
                config.surface_half_life_days,
            )


@dataclass
class PairState:
    player_low_lookup: str
    player_high_lookup: str
    player_low_name: str
    player_high_name: str
    total_matches: int = 0
    low_player_wins: int = 0
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_low_player_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=H2H_RECENT_WINDOW))
    last_match_date: pd.Timestamp | None = None


def ordered_pair_key(player_a_name: str, player_b_name: str) -> tuple[str, str]:
    lookup_a = normalize_player_name(player_a_name)
    lookup_b = normalize_player_name(player_b_name)
    return tuple(sorted((lookup_a, lookup_b)))


def pair_state_to_row(pair_state: PairState) -> dict[str, Any]:
    row = {
        "player_low_lookup": pair_state.player_low_lookup,
        "player_high_lookup": pair_state.player_high_lookup,
        "player_low_name": pair_state.player_low_name,
        "player_high_name": pair_state.player_high_name,
        "total_matches": pair_state.total_matches,
        "low_player_wins": pair_state.low_player_wins,
        "recent_results": ";".join(str(value) for value in pair_state.recent_results),
        "last_match_date": pair_state.last_match_date,
    }
    for surface in SURFACES:
        surface_key = surface.lower()
        row[f"{surface_key}_matches"] = pair_state.surface_matches[surface]
        row[f"{surface_key}_low_player_wins"] = pair_state.surface_low_player_wins[surface]
    return row


def pair_history_frame_from_states(pair_states: dict[tuple[str, str], PairState]) -> pd.DataFrame:
    rows = [pair_state_to_row(pair_state) for pair_state in pair_states.values()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "player_low_lookup",
                "player_high_lookup",
                "player_low_name",
                "player_high_name",
                "total_matches",
                "low_player_wins",
                "recent_results",
                "last_match_date",
            ]
        )
    return pd.DataFrame(rows).sort_values(["player_low_name", "player_high_name"]).reset_index(drop=True)


def parse_pair_state_row(row: dict[str, Any]) -> PairState:
    state = PairState(
        player_low_lookup=str(row["player_low_lookup"]),
        player_high_lookup=str(row["player_high_lookup"]),
        player_low_name=str(row.get("player_low_name") or row["player_low_lookup"]),
        player_high_name=str(row.get("player_high_name") or row["player_high_lookup"]),
        total_matches=int(float(row.get("total_matches", 0) or 0)),
        low_player_wins=int(float(row.get("low_player_wins", 0) or 0)),
    )
    state.recent_results = parse_recent_results(row.get("recent_results"))
    if row.get("last_match_date") not in (None, "", float("nan")) and not pd.isna(row.get("last_match_date")):
        state.last_match_date = pd.Timestamp(row.get("last_match_date"))
    for surface in SURFACES:
        surface_key = surface.lower()
        state.surface_matches[surface] = int(float(row.get(f"{surface_key}_matches", 0) or 0))
        state.surface_low_player_wins[surface] = int(float(row.get(f"{surface_key}_low_player_wins", 0) or 0))
    return state


def pair_history_to_states(pair_history: pd.DataFrame | dict[tuple[str, str], PairState] | None) -> dict[tuple[str, str], PairState]:
    if pair_history is None:
        return {}
    if isinstance(pair_history, dict):
        return pair_history
    states: dict[tuple[str, str], PairState] = {}
    if pair_history.empty:
        return states
    for row in pair_history.to_dict("records"):
        key = (str(row["player_low_lookup"]), str(row["player_high_lookup"]))
        states[key] = parse_pair_state_row(row)
    return states


def pair_history_record(
    pair_states: dict[tuple[str, str], PairState],
    *,
    player_1_name: str,
    player_2_name: str,
    surface: str,
    match_date: pd.Timestamp,
) -> dict[str, float | None]:
    low_lookup, high_lookup = ordered_pair_key(player_1_name, player_2_name)
    pair_state = pair_states.get((low_lookup, high_lookup))
    if pair_state is None or pair_state.total_matches == 0:
        return {
            "h2h_matches": 0.0,
            "h2h_win_rate_edge": 0.0,
            "surface_h2h_matches": 0.0,
            "surface_h2h_win_rate_edge": 0.0,
            "recent_h2h_matches": 0.0,
            "recent_h2h_win_rate_edge": 0.0,
            "days_since_h2h": None,
        }

    player_1_is_low = normalize_player_name(player_1_name) == low_lookup
    player_1_wins = pair_state.low_player_wins if player_1_is_low else pair_state.total_matches - pair_state.low_player_wins
    surface_matches = pair_state.surface_matches[surface]
    surface_low_wins = pair_state.surface_low_player_wins[surface]
    surface_player_1_wins = surface_low_wins if player_1_is_low else surface_matches - surface_low_wins
    recent_matches = len(pair_state.recent_results)
    recent_low_wins = sum(pair_state.recent_results)
    recent_player_1_wins = recent_low_wins if player_1_is_low else recent_matches - recent_low_wins
    days_since_h2h = None
    if pair_state.last_match_date is not None:
        days_since_h2h = float(max(0, (match_date - pair_state.last_match_date).days))
    return {
        "h2h_matches": float(pair_state.total_matches),
        "h2h_win_rate_edge": smooth_win_rate(player_1_wins, pair_state.total_matches) - 0.5,
        "surface_h2h_matches": float(surface_matches),
        "surface_h2h_win_rate_edge": smooth_win_rate(surface_player_1_wins, surface_matches) - 0.5,
        "recent_h2h_matches": float(recent_matches),
        "recent_h2h_win_rate_edge": smooth_win_rate(recent_player_1_wins, recent_matches) - 0.5,
        "days_since_h2h": days_since_h2h,
    }


def update_pair_history(
    pair_states: dict[tuple[str, str], PairState],
    *,
    player_1_name: str,
    player_2_name: str,
    winner_name: str,
    surface: str,
    match_date: pd.Timestamp,
) -> None:
    low_lookup, high_lookup = ordered_pair_key(player_1_name, player_2_name)
    low_name = player_1_name if normalize_player_name(player_1_name) == low_lookup else player_2_name
    high_name = player_2_name if normalize_player_name(player_2_name) == high_lookup else player_1_name
    pair_state = pair_states.setdefault(
        (low_lookup, high_lookup),
        PairState(
            player_low_lookup=low_lookup,
            player_high_lookup=high_lookup,
            player_low_name=low_name,
            player_high_name=high_name,
        ),
    )
    pair_state.player_low_name = low_name
    pair_state.player_high_name = high_name
    pair_state.total_matches += 1
    pair_state.surface_matches[surface] += 1
    low_player_won = normalize_player_name(winner_name) == low_lookup
    pair_state.low_player_wins += int(low_player_won)
    pair_state.surface_low_player_wins[surface] += int(low_player_won)
    pair_state.recent_results.append(1 if low_player_won else 0)
    pair_state.last_match_date = match_date


def serialize_stat_line(stat_line: StatLine) -> str:
    return ",".join(f"{getattr(stat_line, field_name):.6f}" for field_name in STAT_LINE_FIELDS)


def parse_stat_line(value: Any) -> StatLine:
    if pd.isna(value) or value is None or value == "":
        return StatLine()
    items = str(value).split(",")
    values = [float(item) for item in items]
    padded = values + [0.0] * (len(STAT_LINE_FIELDS) - len(values))
    return StatLine(**{field_name: padded[index] for index, field_name in enumerate(STAT_LINE_FIELDS)})


def serialize_recent_stat_lines(stat_lines: deque[StatLine]) -> str:
    return ";".join(serialize_stat_line(stat_line) for stat_line in stat_lines)


def parse_recent_stat_lines(value: Any) -> deque[StatLine]:
    stat_lines: deque[StatLine] = deque(maxlen=RECENT_RESULTS_WINDOW)
    if pd.isna(value) or value is None or value == "":
        return stat_lines
    for item in str(value).split(";"):
        item = item.strip()
        if item:
            stat_lines.append(parse_stat_line(item))
    return stat_lines


def best_of_context_elo(state: PlayerState, best_of: float) -> float:
    if int(best_of) != 5:
        return state.elo
    bo5_weight = state.best_of_five_matches / (state.best_of_five_matches + 8.0)
    return bo5_weight * state.best_of_five_elo + (1.0 - bo5_weight) * state.elo


def pre_match_player_record(
    state: PlayerState,
    surface: str,
    *,
    match_date: pd.Timestamp,
    best_of: float,
) -> dict[str, float | None]:
    days_since_last_match = None
    if state.last_match_date is not None:
        days_since_last_match = float(max(0, (match_date - state.last_match_date).days))

    recent_match_count = float(state.matches_in_recent_window(match_date))
    overall_stats = state.stat_profile()
    surface_stats = state.surface_stat_profile(surface)
    recent_stats = state.recent_stat_profile()
    return {
        "rank": state.last_rank,
        "rank_points": state.last_rank_points,
        "age": state.last_age,
        "height": state.last_height,
        "hand": state.last_hand,
        "matches": float(state.matches),
        "wins": float(state.wins),
        "overall_win_rate": state.overall_win_rate(),
        "surface_matches": float(state.surface_matches[surface]),
        "surface_wins": float(state.surface_wins[surface]),
        "surface_win_rate": state.surface_win_rate(surface),
        "recent_form": state.recent_form(),
        "recent_adjusted_form": state.recent_adjusted_form(),
        "elo": state.elo,
        "recent_elo": state.recent_elo,
        "surface_elo": state.surface_elo[surface],
        "best_of_five_elo": state.best_of_five_elo,
        "best_of_context_elo": best_of_context_elo(state, best_of),
        "best_of_five_matches": float(state.best_of_five_matches),
        "best_of_five_win_rate": state.best_of_five_win_rate(),
        "days_since_last_match": days_since_last_match,
        "matches_last_30_days": recent_match_count,
        "recent_minutes_total": state.recent_minutes_total(),
        "recent_minutes_average": state.recent_minutes_average(),
        "serve_win_rate": overall_stats["serve_win_rate"],
        "return_win_rate": overall_stats["return_win_rate"],
        "first_serve_in_rate": overall_stats["first_serve_in_rate"],
        "first_serve_win_rate": overall_stats["first_serve_win_rate"],
        "second_serve_win_rate": overall_stats["second_serve_win_rate"],
        "ace_rate": overall_stats["ace_rate"],
        "double_fault_rate": overall_stats["double_fault_rate"],
        "bp_save_rate": overall_stats["bp_save_rate"],
        "surface_serve_win_rate": surface_stats["serve_win_rate"],
        "surface_return_win_rate": surface_stats["return_win_rate"],
        "recent_serve_win_rate": recent_stats["serve_win_rate"],
        "recent_return_win_rate": recent_stats["return_win_rate"],
        "utr": state.last_utr,
    }


def build_feature_row(
    player_1: dict[str, float | None],
    player_2: dict[str, float | None],
    *,
    surface: str,
    indoor: str | None = None,
    best_of: float,
    round_name: str,
    tourney_level: str,
    draw_size: float | None,
    matchup: dict[str, float | None] | None = None,
) -> dict[str, float | str | None]:
    matchup = matchup or {
        "h2h_matches": 0.0,
        "h2h_win_rate_edge": 0.0,
        "surface_h2h_matches": 0.0,
        "surface_h2h_win_rate_edge": 0.0,
        "recent_h2h_matches": 0.0,
        "recent_h2h_win_rate_edge": 0.0,
        "days_since_h2h": None,
    }
    player_1_is_lefty = (
        1.0
        if str(player_1.get("hand") or "").upper().startswith("L")
        else 0.0
    )
    player_2_is_lefty = (
        1.0
        if str(player_2.get("hand") or "").upper().startswith("L")
        else 0.0
    )
    serve_vs_return_edge_p1 = player_1["serve_win_rate"] - player_2["return_win_rate"]
    serve_vs_return_edge_p2 = player_2["serve_win_rate"] - player_1["return_win_rate"]
    surface_serve_vs_return_edge_p1 = player_1["surface_serve_win_rate"] - player_2["surface_return_win_rate"]
    surface_serve_vs_return_edge_p2 = player_2["surface_serve_win_rate"] - player_1["surface_return_win_rate"]
    recent_serve_vs_return_edge_p1 = player_1["recent_serve_win_rate"] - player_2["recent_return_win_rate"]
    recent_serve_vs_return_edge_p2 = player_2["recent_serve_win_rate"] - player_1["recent_return_win_rate"]
    second_serve_pressure_p1 = player_1["second_serve_win_rate"] - player_2["return_win_rate"]
    second_serve_pressure_p2 = player_2["second_serve_win_rate"] - player_1["return_win_rate"]
    ace_vs_return_pressure_p1 = player_1["ace_rate"] - player_2["return_win_rate"]
    ace_vs_return_pressure_p2 = player_2["ace_rate"] - player_1["return_win_rate"]
    break_pressure_p1 = player_1["bp_save_rate"] - player_2["return_win_rate"]
    break_pressure_p2 = player_2["bp_save_rate"] - player_1["return_win_rate"]
    surface_form_synergy_p1 = 0.5 * (player_1["surface_win_rate"] + player_1["surface_serve_win_rate"]) + 0.5 * player_1["recent_adjusted_form"]
    surface_form_synergy_p2 = 0.5 * (player_2["surface_win_rate"] + player_2["surface_serve_win_rate"]) + 0.5 * player_2["recent_adjusted_form"]
    recent_surface_form_synergy_p1 = 0.5 * (player_1["recent_serve_win_rate"] + player_1["surface_return_win_rate"]) + 0.5 * player_1["recent_form"]
    recent_surface_form_synergy_p2 = 0.5 * (player_2["recent_serve_win_rate"] + player_2["surface_return_win_rate"]) + 0.5 * player_2["recent_form"]
    return {
        "surface": surface,
        "indoor": None if indoor is None or pd.isna(indoor) or str(indoor).strip() == "" else str(indoor).strip().upper(),
        "best_of": best_of,
        "round": round_name,
        "tourney_level": tourney_level,
        "draw_size": draw_size,
        "player_1_is_lefty": player_1_is_lefty,
        "player_2_is_lefty": player_2_is_lefty,
        "lefty_mismatch": 1.0 if player_1_is_lefty != player_2_is_lefty else 0.0,
        "handedness_matchup": f"{str(player_1.get('hand') or 'U').upper()}_vs_{str(player_2.get('hand') or 'U').upper()}",
        "h2h_matches": matchup["h2h_matches"],
        "h2h_win_rate_edge": matchup["h2h_win_rate_edge"],
        "surface_h2h_matches": matchup["surface_h2h_matches"],
        "surface_h2h_win_rate_edge": matchup["surface_h2h_win_rate_edge"],
        "recent_h2h_matches": matchup["recent_h2h_matches"],
        "recent_h2h_win_rate_edge": matchup["recent_h2h_win_rate_edge"],
        "days_since_h2h": matchup["days_since_h2h"],
        "rank_gap": None
        if player_1["rank"] is None or player_2["rank"] is None
        else player_2["rank"] - player_1["rank"],
        "rank_points_gap": None
        if player_1["rank_points"] is None or player_2["rank_points"] is None
        else player_1["rank_points"] - player_2["rank_points"],
        "age_gap": None
        if player_1["age"] is None or player_2["age"] is None
        else player_1["age"] - player_2["age"],
        "height_gap": None
        if player_1["height"] is None or player_2["height"] is None
        else player_1["height"] - player_2["height"],
        "matches_gap": player_1["matches"] - player_2["matches"],
        "wins_gap": player_1["wins"] - player_2["wins"],
        "overall_win_rate_gap": player_1["overall_win_rate"] - player_2["overall_win_rate"],
        "surface_matches_gap": player_1["surface_matches"] - player_2["surface_matches"],
        "surface_wins_gap": player_1["surface_wins"] - player_2["surface_wins"],
        "surface_win_rate_gap": player_1["surface_win_rate"] - player_2["surface_win_rate"],
        "recent_form_gap": player_1["recent_form"] - player_2["recent_form"],
        "recent_adjusted_form_gap": player_1["recent_adjusted_form"] - player_2["recent_adjusted_form"],
        "elo_gap": player_1["elo"] - player_2["elo"],
        "recent_elo_gap": player_1["recent_elo"] - player_2["recent_elo"],
        "surface_elo_gap": player_1["surface_elo"] - player_2["surface_elo"],
        "best_of_five_elo_gap": player_1["best_of_five_elo"] - player_2["best_of_five_elo"],
        "best_of_context_elo_gap": player_1["best_of_context_elo"] - player_2["best_of_context_elo"],
        "best_of_five_matches_gap": player_1["best_of_five_matches"] - player_2["best_of_five_matches"],
        "best_of_five_win_rate_gap": player_1["best_of_five_win_rate"] - player_2["best_of_five_win_rate"],
        "days_since_last_match_gap": None
        if player_1["days_since_last_match"] is None or player_2["days_since_last_match"] is None
        else player_1["days_since_last_match"] - player_2["days_since_last_match"],
        "matches_last_30_days_gap": player_1["matches_last_30_days"] - player_2["matches_last_30_days"],
        "recent_minutes_total_gap": player_1["recent_minutes_total"] - player_2["recent_minutes_total"],
        "recent_minutes_average_gap": player_1["recent_minutes_average"] - player_2["recent_minutes_average"],
        "serve_win_rate_gap": player_1["serve_win_rate"] - player_2["serve_win_rate"],
        "return_win_rate_gap": player_1["return_win_rate"] - player_2["return_win_rate"],
        "first_serve_in_rate_gap": player_1["first_serve_in_rate"] - player_2["first_serve_in_rate"],
        "first_serve_win_rate_gap": player_1["first_serve_win_rate"] - player_2["first_serve_win_rate"],
        "second_serve_win_rate_gap": player_1["second_serve_win_rate"] - player_2["second_serve_win_rate"],
        "ace_rate_gap": player_1["ace_rate"] - player_2["ace_rate"],
        "double_fault_rate_gap": player_1["double_fault_rate"] - player_2["double_fault_rate"],
        "bp_save_rate_gap": player_1["bp_save_rate"] - player_2["bp_save_rate"],
        "surface_serve_win_rate_gap": player_1["surface_serve_win_rate"] - player_2["surface_serve_win_rate"],
        "surface_return_win_rate_gap": player_1["surface_return_win_rate"] - player_2["surface_return_win_rate"],
        "recent_serve_win_rate_gap": player_1["recent_serve_win_rate"] - player_2["recent_serve_win_rate"],
        "recent_return_win_rate_gap": player_1["recent_return_win_rate"] - player_2["recent_return_win_rate"],
        "serve_vs_return_edge_gap": serve_vs_return_edge_p1 - serve_vs_return_edge_p2,
        "surface_serve_vs_return_edge_gap": surface_serve_vs_return_edge_p1 - surface_serve_vs_return_edge_p2,
        "recent_serve_vs_return_edge_gap": recent_serve_vs_return_edge_p1 - recent_serve_vs_return_edge_p2,
        "second_serve_pressure_gap": second_serve_pressure_p1 - second_serve_pressure_p2,
        "ace_vs_return_pressure_gap": ace_vs_return_pressure_p1 - ace_vs_return_pressure_p2,
        "break_pressure_gap": break_pressure_p1 - break_pressure_p2,
        "surface_form_synergy_gap": surface_form_synergy_p1 - surface_form_synergy_p2,
        "recent_surface_form_synergy_gap": recent_surface_form_synergy_p1 - recent_surface_form_synergy_p2,
        "utr_gap": None
        if player_1["utr"] is None or player_2["utr"] is None
        else player_1["utr"] - player_2["utr"],
    }


def update_player_metadata(
    state: PlayerState,
    name: str,
    rank: Any,
    rank_points: Any,
    age: Any,
    height: Any,
    hand: Any = None,
) -> None:
    state.name = name
    state.last_rank = safe_value(rank)
    state.last_rank_points = safe_value(rank_points)
    state.last_age = safe_value(age)
    state.last_height = safe_value(height)
    if pd.notna(hand) and str(hand).strip():
        state.last_hand = str(hand).strip().upper()


def update_player_result(
    state: PlayerState,
    surface: str,
    *,
    won: bool,
    match_date: pd.Timestamp,
    best_of: float,
    stat_line: StatLine,
    match_minutes: Any = None,
    performance_residual: float | None = None,
) -> None:
    state.matches += 1
    state.wins += int(won)
    state.surface_matches[surface] += 1
    state.surface_wins[surface] += int(won)
    state.recent_results.append(1 if won else 0)
    state.recent_match_dates.append(match_date)
    if performance_residual is not None:
        state.recent_performance_residuals.append(float(performance_residual))
    if pd.notna(match_minutes):
        state.recent_match_minutes.append(float(match_minutes))
    state.stats.add(stat_line)
    state.surface_stats[surface].add(stat_line)
    state.recent_stat_lines.append(stat_line)
    state.matches_in_recent_window(match_date)
    if int(best_of) == 5:
        state.best_of_five_matches += 1
        state.best_of_five_wins += int(won)
    state.last_match_date = match_date


def update_elo(
    state_a: PlayerState,
    state_b: PlayerState,
    *,
    surface: str,
    a_won: bool,
    best_of: float,
    tourney_level: str,
    match_date: pd.Timestamp | None = None,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> None:
    score_a = 1.0 if a_won else 0.0
    score_b = 1.0 - score_a

    level_multiplier = tournament_k_multiplier(tourney_level)
    experience_multiplier = (
        experience_k_multiplier(state_a.matches, config) + experience_k_multiplier(state_b.matches, config)
    ) / 2.0
    inactivity_days_a = 0
    inactivity_days_b = 0
    if match_date is not None:
        if state_a.last_match_date is not None:
            inactivity_days_a = max(0, (match_date - state_a.last_match_date).days)
        if state_b.last_match_date is not None:
            inactivity_days_b = max(0, (match_date - state_b.last_match_date).days)
    inactivity_multiplier = (
        inactivity_k_multiplier(inactivity_days_a, config) + inactivity_k_multiplier(inactivity_days_b, config)
    ) / 2.0
    best_of_multiplier = config.best_of_five_match_multiplier if int(best_of) == 5 else 1.0
    k = config.base_k * level_multiplier * experience_multiplier * inactivity_multiplier * best_of_multiplier

    expected_a = expected_score(state_a.elo, state_b.elo)
    expected_b = expected_score(state_b.elo, state_a.elo)
    state_a.elo += k * (score_a - expected_a)
    state_b.elo += k * (score_b - expected_b)

    expected_recent_a = expected_score(state_a.recent_elo, state_b.recent_elo)
    expected_recent_b = expected_score(state_b.recent_elo, state_a.recent_elo)
    state_a.recent_elo += (k * config.recent_k_multiplier) * (score_a - expected_recent_a)
    state_b.recent_elo += (k * config.recent_k_multiplier) * (score_b - expected_recent_b)

    surface_a = state_a.surface_elo[surface]
    surface_b = state_b.surface_elo[surface]
    expected_surface_a = expected_score(surface_a, surface_b)
    expected_surface_b = expected_score(surface_b, surface_a)
    state_a.surface_elo[surface] = surface_a + (k * config.surface_k_multiplier) * (score_a - expected_surface_a)
    state_b.surface_elo[surface] = surface_b + (k * config.surface_k_multiplier) * (score_b - expected_surface_b)

    if int(best_of) == 5:
        expected_bo5_a = expected_score(state_a.best_of_five_elo, state_b.best_of_five_elo)
        expected_bo5_b = expected_score(state_b.best_of_five_elo, state_a.best_of_five_elo)
        state_a.best_of_five_elo += (k * config.best_of_five_k_multiplier) * (score_a - expected_bo5_a)
        state_b.best_of_five_elo += (k * config.best_of_five_k_multiplier) * (score_b - expected_bo5_b)


def player_state_to_snapshot_row(
    player_key_value: str,
    state: PlayerState,
    utr_tracker: Any | None = None,
) -> dict[str, Any]:
    overall_stats = state.stat_profile()
    recent_stats = state.recent_stat_profile()
    return {
        "player_key": player_key_value,
        "player_name": state.name,
        "last_rank": state.last_rank,
        "last_rank_points": state.last_rank_points,
        "last_age": state.last_age,
        "last_height": state.last_height,
        "last_hand": state.last_hand,
        "matches": state.matches,
        "wins": state.wins,
        "overall_win_rate": state.overall_win_rate(),
        "recent_form": state.recent_form(),
        "elo": state.elo,
        "recent_elo": state.recent_elo,
        "best_of_five_elo": state.best_of_five_elo,
        "best_of_five_matches": state.best_of_five_matches,
        "best_of_five_wins": state.best_of_five_wins,
        "best_of_five_win_rate": state.best_of_five_win_rate(),
        "last_match_date": state.last_match_date,
        "recent_match_dates": ";".join(
            timestamp.strftime("%Y-%m-%d") for timestamp in state.recent_match_dates
        ),
        "recent_results": ";".join(str(value) for value in state.recent_results),
        "recent_performance_residuals": serialize_float_deque(state.recent_performance_residuals),
        "recent_match_minutes": serialize_float_deque(state.recent_match_minutes),
        "serve_win_rate": overall_stats["serve_win_rate"],
        "return_win_rate": overall_stats["return_win_rate"],
        "first_serve_in_rate": overall_stats["first_serve_in_rate"],
        "first_serve_win_rate": overall_stats["first_serve_win_rate"],
        "second_serve_win_rate": overall_stats["second_serve_win_rate"],
        "ace_rate": overall_stats["ace_rate"],
        "double_fault_rate": overall_stats["double_fault_rate"],
        "bp_save_rate": overall_stats["bp_save_rate"],
        "recent_serve_win_rate": recent_stats["serve_win_rate"],
        "recent_return_win_rate": recent_stats["return_win_rate"],
        "utr_singles": state.last_utr if utr_tracker is None else utr_tracker.latest_rating(state.name),
        **{f"{surface.lower()}_matches": state.surface_matches[surface] for surface in SURFACES},
        **{f"{surface.lower()}_wins": state.surface_wins[surface] for surface in SURFACES},
        **{f"{surface.lower()}_win_rate": state.surface_win_rate(surface) for surface in SURFACES},
        **{f"{surface.lower()}_elo": state.surface_elo[surface] for surface in SURFACES},
        **{
            f"{surface.lower()}_serve_win_rate": state.surface_stat_profile(surface)["serve_win_rate"]
            for surface in SURFACES
        },
        **{
            f"{surface.lower()}_return_win_rate": state.surface_stat_profile(surface)["return_win_rate"]
            for surface in SURFACES
        },
    }


def player_state_to_live_row(
    player_key_value: str,
    state: PlayerState,
) -> dict[str, Any]:
    row = {
        "player_key": player_key_value,
        "player_name": state.name,
        "last_rank": state.last_rank,
        "last_rank_points": state.last_rank_points,
        "last_age": state.last_age,
        "last_height": state.last_height,
        "last_hand": state.last_hand,
        "matches": state.matches,
        "wins": state.wins,
        "elo": state.elo,
        "recent_elo": state.recent_elo,
        "best_of_five_elo": state.best_of_five_elo,
        "best_of_five_matches": state.best_of_five_matches,
        "best_of_five_wins": state.best_of_five_wins,
        "last_match_date": state.last_match_date,
        "recent_match_dates": ";".join(
            timestamp.strftime("%Y-%m-%d") for timestamp in state.recent_match_dates
        ),
        "recent_results": ";".join(str(value) for value in state.recent_results),
        "recent_performance_residuals": serialize_float_deque(state.recent_performance_residuals),
        "recent_match_minutes": serialize_float_deque(state.recent_match_minutes),
        "recent_stat_lines": serialize_recent_stat_lines(state.recent_stat_lines),
        "overall_stat_line": serialize_stat_line(state.stats),
        "utr_singles": state.last_utr,
    }
    for surface in SURFACES:
        surface_key = surface.lower()
        row[f"{surface_key}_matches"] = state.surface_matches[surface]
        row[f"{surface_key}_wins"] = state.surface_wins[surface]
        row[f"{surface_key}_elo"] = state.surface_elo[surface]
        row[f"{surface_key}_stat_line"] = serialize_stat_line(state.surface_stats[surface])
    return row


def snapshot_and_live_state_from_states(
    states: dict[str, PlayerState],
    *,
    utr_tracker: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot_rows = [
        player_state_to_snapshot_row(player_key_value, state, utr_tracker=utr_tracker)
        for player_key_value, state in states.items()
    ]
    live_rows = [
        player_state_to_live_row(player_key_value, state)
        for player_key_value, state in states.items()
    ]
    snapshot = pd.DataFrame(snapshot_rows).sort_values("player_name").reset_index(drop=True)
    live_state = pd.DataFrame(live_rows).sort_values("player_name").reset_index(drop=True)
    return snapshot, live_state


def build_training_frame_with_state(
    matches: pd.DataFrame,
    utr_tracker: Any | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states: dict[str, PlayerState] = {}
    pair_states: dict[tuple[str, str], PairState] = {}
    rows: list[dict[str, Any]] = []

    for match in matches.itertuples(index=False):
        surface = str(match.surface)
        best_of = float(match.best_of)
        match_date = match.tourney_date
        winner_key = player_key(match.winner_id, match.winner_name)
        loser_key = player_key(match.loser_id, match.loser_name)

        winner_state = states.setdefault(winner_key, PlayerState(name=match.winner_name))
        loser_state = states.setdefault(loser_key, PlayerState(name=match.loser_name))

        winner_state.apply_decay(match_date, config=elo_config)
        loser_state.apply_decay(match_date, config=elo_config)

        update_player_metadata(
            winner_state,
            match.winner_name,
            match.winner_rank,
            match.winner_rank_points,
            match.winner_age,
            match.winner_ht,
            getattr(match, "winner_hand", None),
        )
        update_player_metadata(
            loser_state,
            match.loser_name,
            match.loser_rank,
            match.loser_rank_points,
            match.loser_age,
            match.loser_ht,
            getattr(match, "loser_hand", None),
        )
        if utr_tracker is not None:
            winner_state.last_utr = utr_tracker.rating_for_player_on_date(
                match.winner_name,
                match_date,
            )
            loser_state.last_utr = utr_tracker.rating_for_player_on_date(
                match.loser_name,
                match_date,
            )

        winner_stat_line = build_match_stat_line(
            ace=match.w_ace,
            double_fault=match.w_df,
            serve_points_total=match.w_svpt,
            first_serves_in=match.w_1stIn,
            first_serve_points_won=match.w_1stWon,
            second_serve_points_won=match.w_2ndWon,
            break_points_saved=match.w_bpSaved,
            break_points_faced=match.w_bpFaced,
            opponent_serve_points_total=match.l_svpt,
            opponent_first_serve_points_won=match.l_1stWon,
            opponent_second_serve_points_won=match.l_2ndWon,
        )
        loser_stat_line = build_match_stat_line(
            ace=match.l_ace,
            double_fault=match.l_df,
            serve_points_total=match.l_svpt,
            first_serves_in=match.l_1stIn,
            first_serve_points_won=match.l_1stWon,
            second_serve_points_won=match.l_2ndWon,
            break_points_saved=match.l_bpSaved,
            break_points_faced=match.l_bpFaced,
            opponent_serve_points_total=match.w_svpt,
            opponent_first_serve_points_won=match.w_1stWon,
            opponent_second_serve_points_won=match.w_2ndWon,
        )

        winner_features = pre_match_player_record(
            winner_state,
            surface,
            match_date=match_date,
            best_of=best_of,
        )
        loser_features = pre_match_player_record(
            loser_state,
            surface,
            match_date=match_date,
            best_of=best_of,
        )

        base_kwargs = {
            "surface": surface,
            "indoor": getattr(match, "indoor", None),
            "best_of": best_of,
            "round_name": str(match.round),
            "tourney_level": str(match.tourney_level),
            "draw_size": safe_value(match.draw_size),
        }
        matchup_winner = pair_history_record(
            pair_states,
            player_1_name=match.winner_name,
            player_2_name=match.loser_name,
            surface=surface,
            match_date=match_date,
        )
        matchup_loser = pair_history_record(
            pair_states,
            player_1_name=match.loser_name,
            player_2_name=match.winner_name,
            surface=surface,
            match_date=match_date,
        )

        row_win = build_feature_row(winner_features, loser_features, matchup=matchup_winner, **base_kwargs)
        row_win["target"] = 1
        row_win["match_date"] = match_date
        row_win["player_1"] = match.winner_name
        row_win["player_2"] = match.loser_name
        rows.append(row_win)

        row_loss = build_feature_row(loser_features, winner_features, matchup=matchup_loser, **base_kwargs)
        row_loss["target"] = 0
        row_loss["match_date"] = match_date
        row_loss["player_1"] = match.loser_name
        row_loss["player_2"] = match.winner_name
        rows.append(row_loss)

        winner_expected = expected_score(winner_state.elo, loser_state.elo)
        loser_expected = 1.0 - winner_expected

        update_elo(
            winner_state,
            loser_state,
            surface=surface,
            a_won=True,
            best_of=best_of,
            tourney_level=str(match.tourney_level),
            match_date=match_date,
            config=elo_config,
        )
        update_player_result(
            winner_state,
            surface,
            won=True,
            match_date=match_date,
            best_of=best_of,
            stat_line=winner_stat_line,
            match_minutes=getattr(match, "minutes", None),
            performance_residual=1.0 - winner_expected,
        )
        update_player_result(
            loser_state,
            surface,
            won=False,
            match_date=match_date,
            best_of=best_of,
            stat_line=loser_stat_line,
            match_minutes=getattr(match, "minutes", None),
            performance_residual=0.0 - loser_expected,
        )
        update_pair_history(
            pair_states,
            player_1_name=match.winner_name,
            player_2_name=match.loser_name,
            winner_name=match.winner_name,
            surface=surface,
            match_date=match_date,
        )

    training_frame = pd.DataFrame(rows).sort_values(
        ["match_date", "player_1", "player_2"]
    ).reset_index(drop=True)

    snapshot, live_state = snapshot_and_live_state_from_states(
        states,
        utr_tracker=utr_tracker,
    )
    pair_history = pair_history_frame_from_states(pair_states)
    return training_frame, snapshot, live_state, pair_history


def build_training_frame(
    matches: pd.DataFrame,
    utr_tracker: Any | None = None,
    elo_config: EloConfig = DEFAULT_ELO_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_frame, snapshot, _, _ = build_training_frame_with_state(
        matches,
        utr_tracker=utr_tracker,
        elo_config=elo_config,
    )
    return training_frame, snapshot


def parse_recent_match_dates(value: Any) -> deque[pd.Timestamp]:
    dates: deque[pd.Timestamp] = deque()
    if pd.isna(value) or value is None or value == "":
        return dates
    for item in str(value).split(";"):
        item = item.strip()
        if item:
            dates.append(pd.Timestamp(item))
    return dates


def parse_recent_results(value: Any) -> deque[int]:
    results: deque[int] = deque(maxlen=RECENT_RESULTS_WINDOW)
    if pd.isna(value) or value is None or value == "":
        return results
    for item in str(value).split(";"):
        item = item.strip()
        if item:
            results.append(int(item))
    return results


def state_from_snapshot_row(snapshot_row: dict[str, Any], player_name: str) -> PlayerState:
    state = PlayerState(name=player_name)
    state.last_rank = safe_value(snapshot_row.get("last_rank"))
    state.last_rank_points = safe_value(snapshot_row.get("last_rank_points"))
    state.last_age = safe_value(snapshot_row.get("last_age"))
    state.last_height = safe_value(snapshot_row.get("last_height"))
    if pd.notna(snapshot_row.get("last_hand")) and str(snapshot_row.get("last_hand")).strip():
        state.last_hand = str(snapshot_row.get("last_hand")).strip().upper()
    state.matches = int(float(snapshot_row.get("matches", 0) or 0))
    state.wins = int(float(snapshot_row.get("wins", 0) or 0))
    state.elo = float(snapshot_row.get("elo", BASE_ELO))
    state.recent_elo = float(snapshot_row.get("recent_elo", BASE_ELO))
    state.best_of_five_elo = float(snapshot_row.get("best_of_five_elo", BASE_ELO))
    state.best_of_five_matches = int(float(snapshot_row.get("best_of_five_matches", 0) or 0))
    state.best_of_five_wins = int(float(snapshot_row.get("best_of_five_wins", 0) or 0))
    state.last_utr = safe_value(snapshot_row.get("utr_singles"))
    if snapshot_row.get("last_match_date") not in (None, "", float("nan")) and not pd.isna(snapshot_row.get("last_match_date")):
        state.last_match_date = pd.Timestamp(snapshot_row.get("last_match_date"))
    state.recent_match_dates = parse_recent_match_dates(snapshot_row.get("recent_match_dates"))
    state.recent_results = parse_recent_results(snapshot_row.get("recent_results"))
    state.recent_performance_residuals = parse_float_deque(snapshot_row.get("recent_performance_residuals"))
    state.recent_match_minutes = parse_float_deque(snapshot_row.get("recent_match_minutes"))

    for surface in SURFACES:
        surface_key = surface.lower()
        state.surface_matches[surface] = int(float(snapshot_row.get(f"{surface_key}_matches", 0) or 0))
        state.surface_wins[surface] = int(float(snapshot_row.get(f"{surface_key}_wins", 0) or 0))
        state.surface_elo[surface] = float(snapshot_row.get(f"{surface_key}_elo", BASE_ELO))
    return state


def state_from_live_state_row(live_state_row: dict[str, Any], player_name: str) -> PlayerState:
    state = state_from_snapshot_row(live_state_row, player_name)
    state.stats = parse_stat_line(live_state_row.get("overall_stat_line"))
    state.recent_stat_lines = parse_recent_stat_lines(live_state_row.get("recent_stat_lines"))
    for surface in SURFACES:
        state.surface_stats[surface] = parse_stat_line(
            live_state_row.get(f"{surface.lower()}_stat_line")
        )
    return state


def snapshot_from_live_state(live_state: pd.DataFrame) -> pd.DataFrame:
    states: dict[str, PlayerState] = {}
    for row in live_state.to_dict("records"):
        player_name = str(row.get("player_name", "Unknown Player"))
        key = str(row.get("player_key") or player_key(None, player_name))
        states[key] = state_from_live_state_row(row, player_name)
    snapshot, _ = snapshot_and_live_state_from_states(states)
    return snapshot


def snapshot_player_record(
    snapshot_row: dict[str, Any],
    surface: str,
    *,
    match_date: pd.Timestamp,
    best_of: float,
) -> dict[str, float | None]:
    player_name = str(snapshot_row.get("player_name", "Unknown Player"))
    state = state_from_snapshot_row(snapshot_row, player_name)
    state.apply_decay(match_date)
    return pre_match_player_record(
        state,
        surface,
        match_date=match_date,
        best_of=best_of,
    ) | {
        "serve_win_rate": float(snapshot_row.get("serve_win_rate", 0.62)),
        "return_win_rate": float(snapshot_row.get("return_win_rate", 0.38)),
        "first_serve_in_rate": float(snapshot_row.get("first_serve_in_rate", 0.60)),
        "first_serve_win_rate": float(snapshot_row.get("first_serve_win_rate", 0.70)),
        "second_serve_win_rate": float(snapshot_row.get("second_serve_win_rate", 0.50)),
        "ace_rate": float(snapshot_row.get("ace_rate", 0.05)),
        "double_fault_rate": float(snapshot_row.get("double_fault_rate", 0.03)),
        "bp_save_rate": float(snapshot_row.get("bp_save_rate", 0.60)),
        "surface_serve_win_rate": float(snapshot_row.get(f"{surface.lower()}_serve_win_rate", 0.62)),
        "surface_return_win_rate": float(snapshot_row.get(f"{surface.lower()}_return_win_rate", 0.38)),
        "recent_serve_win_rate": float(snapshot_row.get("recent_serve_win_rate", 0.62)),
        "recent_return_win_rate": float(snapshot_row.get("recent_return_win_rate", 0.38)),
    }


def default_snapshot_row(player_name: str) -> dict[str, Any]:
    row = {
        "player_name": player_name,
        "last_rank": None,
        "last_rank_points": None,
        "last_age": None,
        "last_height": None,
        "last_hand": None,
        "matches": 0,
        "wins": 0,
        "overall_win_rate": 0.5,
        "recent_form": 0.5,
        "elo": BASE_ELO,
        "recent_elo": BASE_ELO,
        "best_of_five_elo": BASE_ELO,
        "best_of_five_matches": 0,
        "best_of_five_wins": 0,
        "best_of_five_win_rate": 0.5,
        "last_match_date": None,
        "recent_match_dates": "",
        "recent_results": "",
        "recent_performance_residuals": "",
        "recent_match_minutes": "",
        "serve_win_rate": 0.62,
        "return_win_rate": 0.38,
        "first_serve_in_rate": 0.60,
        "first_serve_win_rate": 0.70,
        "second_serve_win_rate": 0.50,
        "ace_rate": 0.05,
        "double_fault_rate": 0.03,
        "bp_save_rate": 0.60,
        "recent_serve_win_rate": 0.62,
        "recent_return_win_rate": 0.38,
        "utr_singles": None,
    }
    for surface in SURFACES:
        surface_key = surface.lower()
        row[f"{surface_key}_matches"] = 0
        row[f"{surface_key}_wins"] = 0
        row[f"{surface_key}_win_rate"] = 0.5
        row[f"{surface_key}_elo"] = BASE_ELO
        row[f"{surface_key}_serve_win_rate"] = 0.62
        row[f"{surface_key}_return_win_rate"] = 0.38
    return row
