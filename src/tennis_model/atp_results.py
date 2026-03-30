from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from tennis_model.names import normalize_player_name


ATP_STATS_API_TEMPLATE = (
    "https://itp-atp-sls.infosys-platforms.com/prod/api/stats-plus/v1/keystats/"
    "year/{year}/eventId/{event_id}/matchId/{match_id}"
)
BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass
class ATPCompletedMatch:
    match_id: str
    match_date: str | None
    round_text: str
    round_name: str
    court_label: str | None
    duration_text: str | None
    player_1: str
    player_2: str
    winner: str
    score_text: str | None
    results_url: str
    stats_url: str | None
    h2h_url: str | None
    stats_fetch_error: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "match_date": self.match_date,
            "round_text": self.round_text,
            "round_name": self.round_name,
            "court_label": self.court_label,
            "duration_text": self.duration_text,
            "player_1": self.player_1,
            "player_2": self.player_2,
            "winner": self.winner,
            "score_text": self.score_text,
            "results_url": self.results_url,
            "stats_url": self.stats_url,
            "h2h_url": self.h2h_url,
            "stats_fetch_error": self.stats_fetch_error,
        }


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _with_query_params(url: str, **params: str | None) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    for key, value in params.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = [value]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _normalize_round_label(round_text: str) -> str:
    cleaned = _clean_text(round_text)
    lowered = cleaned.lower()
    qualifying_map = {
        "1st round qualifying": "Q1",
        "2nd round qualifying": "Q2",
        "final round qualifying": "Q3",
    }
    if lowered in qualifying_map:
        return qualifying_map[lowered]
    if lowered in {"final", "finals"}:
        return "F"
    if lowered in {"semi-finals", "semifinals", "semi finals"}:
        return "SF"
    if lowered in {"quarter-finals", "quarterfinals", "quarter finals"}:
        return "QF"
    if lowered == "round robin":
        return "RR"
    if lowered.startswith("round of "):
        suffix = cleaned.split()[-1]
        return f"R{suffix}"
    return cleaned


def results_url_from_draw_url(draw_url: str) -> str:
    parsed = urlparse(draw_url)
    path_parts = parsed.path.split("/")
    for index in range(len(path_parts) - 1, -1, -1):
        if path_parts[index] == "draws":
            path_parts[index] = "results"
            return urlunparse(parsed._replace(path="/".join(path_parts), query=""))
    raise ValueError(f"Could not derive ATP results URL from draw URL: {draw_url}")


def _parse_match_dates(soup: BeautifulSoup) -> list[str]:
    options = soup.select('.atp_filters-dropdown[data-key="matchDate"] option')
    date_values = [_clean_text(option.get("value")) for option in options]
    return [value for value in date_values if value]


def _player_name_from_stats_item(stats_item: Tag) -> str:
    name_anchor = stats_item.select_one(".player-info .name a")
    if name_anchor is not None:
        return _clean_text(name_anchor.get_text(" ", strip=True))
    name_node = stats_item.select_one(".player-info .name")
    return _clean_text(name_node.get_text(" ", strip=True) if name_node is not None else "")


def _player_is_winner(stats_item: Tag) -> bool:
    return stats_item.select_one(".player-info .winner") is not None


def _score_text_from_stats_item(stats_item: Tag) -> str | None:
    score_parts = [
        _clean_text(score_node.get_text(" ", strip=True))
        for score_node in stats_item.select(".scores .score-item")
    ]
    score_parts = [part for part in score_parts if part]
    if not score_parts:
        return None
    return " ".join(score_parts)


def _extract_match_id(stats_url: str | None) -> str:
    if not stats_url:
        return ""
    path_parts = [part for part in urlparse(stats_url).path.split("/") if part]
    return path_parts[-1].upper() if path_parts else ""


def _winner_from_notes(notes_text: str, player_1: str, player_2: str) -> str | None:
    normalized_notes = _clean_text(notes_text)
    if not normalized_notes:
        return None
    candidate_names: list[str] = []
    for pattern in (
        r"Game Set and Match (?P<name>.+?)\.",
        r"(?P<name>.+?) wins the match",
    ):
        match = re.search(pattern, normalized_notes, flags=re.IGNORECASE)
        if match is not None:
            candidate_names.append(_clean_text(match.group("name")))

    for candidate_name in candidate_names:
        normalized_candidate = normalize_player_name(candidate_name)
        if normalized_candidate == normalize_player_name(player_1):
            return player_1
        if normalized_candidate == normalize_player_name(player_2):
            return player_2
    return None


def _parse_completed_match_card(match_card: Tag, *, results_url: str, match_date: str | None) -> ATPCompletedMatch | None:
    header_text = _clean_text(
        match_card.select_one(".match-header strong").get_text(" ", strip=True)
        if match_card.select_one(".match-header strong") is not None
        else ""
    )
    if not header_text:
        return None

    header_parts = [part.strip() for part in header_text.split(" - ", 1)]
    round_text = header_parts[0]
    court_label = header_parts[1] if len(header_parts) > 1 else None
    round_name = _normalize_round_label(round_text)

    duration_spans = match_card.select(".match-header span")
    duration_text = _clean_text(duration_spans[-1].get_text(" ", strip=True)) if len(duration_spans) >= 2 else None

    player_rows = match_card.select(".match-content .stats-item")
    if len(player_rows) < 2:
        return None

    player_1 = _player_name_from_stats_item(player_rows[0])
    player_2 = _player_name_from_stats_item(player_rows[1])
    if not player_1 or not player_2:
        return None

    player_1_winner = _player_is_winner(player_rows[0])
    player_2_winner = _player_is_winner(player_rows[1])
    if player_1_winner and not player_2_winner:
        winner = player_1
    elif player_2_winner and not player_1_winner:
        winner = player_2
    else:
        notes_text = _clean_text(
            match_card.select_one(".match-notes").get_text(" ", strip=True)
            if match_card.select_one(".match-notes") is not None
            else ""
        )
        winner = _winner_from_notes(notes_text, player_1, player_2)
        if winner is None:
            return None

    links = {
        _clean_text(link.get_text(" ", strip=True)).lower(): urljoin(results_url, link.get("href", ""))
        for link in match_card.select(".match-footer a")
        if link.get("href")
    }
    stats_url = links.get("stats")
    h2h_url = links.get("h2h")
    match_id = _extract_match_id(stats_url)
    if not match_id:
        return None

    score_1 = _score_text_from_stats_item(player_rows[0])
    score_2 = _score_text_from_stats_item(player_rows[1])
    score_text = None
    if score_1 or score_2:
        score_text = f"{player_1}: {score_1 or '-'} | {player_2}: {score_2 or '-'}"

    return ATPCompletedMatch(
        match_id=match_id,
        match_date=match_date,
        round_text=round_text,
        round_name=round_name,
        court_label=court_label,
        duration_text=duration_text,
        player_1=player_1,
        player_2=player_2,
        winner=winner,
        score_text=score_text,
        results_url=results_url,
        stats_url=stats_url,
        h2h_url=h2h_url,
    )


def _parse_atp_results_page(
    html: str,
    *,
    results_url: str,
    match_date: str | None,
    allowed_rounds: set[str] | None,
) -> list[ATPCompletedMatch]:
    soup = BeautifulSoup(html, "lxml")
    matches: list[ATPCompletedMatch] = []
    for match_card in soup.select(".match-group .match"):
        parsed_match = _parse_completed_match_card(
            match_card,
            results_url=results_url,
            match_date=match_date,
        )
        if parsed_match is None:
            continue
        if allowed_rounds is not None and parsed_match.round_name not in allowed_rounds:
            continue
        matches.append(parsed_match)
    return matches


def _to_base(number: int, base: int) -> str:
    if number == 0:
        return "0"
    digits = []
    while number:
        number, remainder = divmod(number, base)
        digits.append(BASE36_ALPHABET[remainder])
    return "".join(reversed(digits))


def _stats_key_from_last_modified(last_modified_ms: int) -> str:
    local_timestamp = datetime.fromtimestamp(last_modified_ms / 1000)
    timezone_offset_minutes = -int(datetime.now().astimezone().utcoffset().total_seconds() // 60)
    utc_day = datetime.fromtimestamp(
        (last_modified_ms + 60 * timezone_offset_minutes * 1000) / 1000
    ).day
    reversed_day = int(str(utc_day).zfill(2)[::-1])
    year = local_timestamp.year
    reversed_year = int(str(year)[::-1])
    raw_key = _to_base(int(str(last_modified_ms), 16), 36)
    raw_key += _to_base((year + reversed_year) * (utc_day + reversed_day), 24)
    raw_key = (raw_key + ("0" * 14))[:14]
    return f"#{raw_key}$"


def decrypt_stats_payload(payload: dict[str, Any]) -> dict[str, Any]:
    key = _stats_key_from_last_modified(int(payload["lastModified"]))
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv=key.upper().encode("utf-8"))
    decrypted = cipher.decrypt(base64.b64decode(payload["response"]))
    return json.loads(unpad(decrypted, AES.block_size).decode("utf-8"))


def _parse_stats_url(stats_url: str) -> tuple[str, str, str]:
    path_parts = [part for part in urlparse(stats_url).path.split("/") if part]
    if len(path_parts) < 3:
        raise ValueError(f"Could not parse ATP stats URL: {stats_url}")
    year = path_parts[-3]
    event_id = path_parts[-2]
    match_id = path_parts[-1].upper()
    return year, event_id, match_id


def _parse_integer_stat(raw_value: str | None) -> int | None:
    cleaned = _clean_text(raw_value)
    if not cleaned:
        return None
    match = re.search(r"-?\d+", cleaned)
    return int(match.group()) if match else None


def _parse_fraction_stat(raw_value: str | None) -> tuple[int | None, int | None]:
    cleaned = _clean_text(raw_value)
    if not cleaned:
        return None, None
    match = re.search(r"(\d+)\s*/\s*(\d+)", cleaned)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _stats_summary_lookup(stats_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = stats_payload.get("setStats", {}).get("set0", [])
    return {
        _clean_text(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and _clean_text(row.get("name"))
    }


def _stats_for_side(summary_rows: dict[str, dict[str, Any]], *, player_column: str, opponent_column: str) -> dict[str, Any]:
    first_serve_in, serve_points_total = _parse_fraction_stat(
        summary_rows.get("1st Serve", {}).get(player_column)
    )
    first_serve_points_won, _ = _parse_fraction_stat(
        summary_rows.get("1st Serve Points Won", {}).get(player_column)
    )
    second_serve_points_won, _ = _parse_fraction_stat(
        summary_rows.get("2nd Serve Points Won", {}).get(player_column)
    )
    break_points_saved, break_points_faced = _parse_fraction_stat(
        summary_rows.get("Break Points Saved", {}).get(player_column)
    )
    opponent_first_serve_points_won, _ = _parse_fraction_stat(
        summary_rows.get("1st Serve Points Won", {}).get(opponent_column)
    )
    opponent_second_serve_points_won, _ = _parse_fraction_stat(
        summary_rows.get("2nd Serve Points Won", {}).get(opponent_column)
    )
    _, opponent_serve_points_total = _parse_fraction_stat(
        summary_rows.get("1st Serve", {}).get(opponent_column)
    )
    return {
        "ace": _parse_integer_stat(summary_rows.get("Aces", {}).get(player_column)),
        "df": _parse_integer_stat(summary_rows.get("Double Faults", {}).get(player_column)),
        "svpt": serve_points_total,
        "1stIn": first_serve_in,
        "1stWon": first_serve_points_won,
        "2ndWon": second_serve_points_won,
        "bpSaved": break_points_saved,
        "bpFaced": break_points_faced,
        "opp_svpt": opponent_serve_points_total,
        "opp_1stWon": opponent_first_serve_points_won,
        "opp_2ndWon": opponent_second_serve_points_won,
    }


def _payload_player_names(stats_payload: dict[str, Any]) -> list[str]:
    players = stats_payload.get("players", [])
    names: list[str] = []
    for player in players:
        first_name = _clean_text(player.get("player1FirstName"))
        last_name = _clean_text(str(player.get("player1LastName") or "").title())
        full_name = _clean_text(f"{first_name} {last_name}")
        names.append(full_name or _clean_text(player.get("player1Name")))
    return names


def extract_match_stat_columns(
    stats_payload: dict[str, Any],
    *,
    player_1: str,
    player_2: str,
) -> dict[str, Any]:
    summary_rows = _stats_summary_lookup(stats_payload)
    player_1_stats = _stats_for_side(summary_rows, player_column="player1", opponent_column="player2")
    player_2_stats = _stats_for_side(summary_rows, player_column="player2", opponent_column="player1")

    payload_names = _payload_player_names(stats_payload)
    if len(payload_names) >= 2:
        payload_first = normalize_player_name(payload_names[0])
        payload_second = normalize_player_name(payload_names[1])
        lookup_1 = normalize_player_name(player_1)
        lookup_2 = normalize_player_name(player_2)
        if payload_first == lookup_2 and payload_second == lookup_1:
            player_1_stats, player_2_stats = player_2_stats, player_1_stats

    columns: dict[str, Any] = {}
    for stat_name, value in player_1_stats.items():
        if stat_name.startswith("opp_"):
            continue
        columns[f"p1_{stat_name}"] = value
    for stat_name, value in player_2_stats.items():
        if stat_name.startswith("opp_"):
            continue
        columns[f"p2_{stat_name}"] = value
    return columns


def fetch_match_stats(
    stats_url: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    year, event_id, match_id = _parse_stats_url(stats_url)
    api_url = ATP_STATS_API_TEMPLATE.format(year=year, event_id=event_id, match_id=match_id)
    active_session = session or requests.Session()
    response = active_session.get(api_url, timeout=timeout)
    response.raise_for_status()
    return decrypt_stats_payload(response.json())


def fetch_atp_completed_results(
    results_url: str,
    *,
    match_type: str = "singles",
    allowed_rounds: set[str] | None = None,
    surface: str,
    best_of: int,
    tourney_level: str,
    timeout: int = 30,
) -> pd.DataFrame:
    active_session = requests.Session()
    base_results_url = _with_query_params(results_url, matchType=match_type, matchDate=None)
    response = active_session.get(base_results_url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    date_values = _parse_match_dates(soup)
    if not date_values:
        date_values = [None]

    matches_by_id: dict[str, ATPCompletedMatch] = {}
    for match_date in date_values:
        page_url = _with_query_params(base_results_url, matchDate=match_date)
        page_response = active_session.get(page_url, timeout=timeout)
        page_response.raise_for_status()
        for match in _parse_atp_results_page(
            page_response.text,
            results_url=page_url,
            match_date=match_date,
            allowed_rounds=allowed_rounds,
        ):
            matches_by_id[match.match_id] = match

    rows: list[dict[str, Any]] = []
    for match_id in sorted(matches_by_id):
        match = matches_by_id[match_id]
        row = match.to_row()
        row["surface"] = surface
        row["best_of"] = best_of
        row["tourney_level"] = tourney_level
        if match.stats_url:
            try:
                stats_payload = fetch_match_stats(
                    match.stats_url,
                    session=active_session,
                    timeout=timeout,
                )
                row.update(
                    extract_match_stat_columns(
                        stats_payload,
                        player_1=match.player_1,
                        player_2=match.player_2,
                    )
                )
            except Exception as exc:
                row["stats_fetch_error"] = str(exc)
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "match_date",
                "round_text",
                "round_name",
                "court_label",
                "duration_text",
                "player_1",
                "player_2",
                "winner",
                "score_text",
                "results_url",
                "stats_url",
                "h2h_url",
                "surface",
                "best_of",
                "tourney_level",
                "stats_fetch_error",
            ]
        )

    frame = pd.DataFrame(rows)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.date.astype("string")
    sort_columns = [column for column in ["match_date", "round_name", "player_1", "player_2"] if column in frame.columns]
    return frame.sort_values(sort_columns, na_position="last").reset_index(drop=True)
