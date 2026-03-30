from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_SLAMS = ("wimbledon", "us_open", "australian_open", "roland_garros")
DEFAULT_RG_EVENT_CODES = ("SM",)
AO_RESULTS_URL = "https://ausopen.com/results"
RG_RESULTS_URL_TEMPLATE = "https://www.rolandgarros.com/en-us/results/{event_code}"
RG_BASE_URL = "https://www.rolandgarros.com"
WIMBLEDON_CONFIG_URL = "https://www.wimbledon.com/en_GB/json/gen/config_web.json"
US_OPEN_CONFIG_URL = "https://www.usopen.org/en_US/json/gen/config_web.json"


@dataclass
class GrandSlamMatchSnapshot:
    slam_code: str
    provider_name: str
    source_match_id: str
    season_year: int | None
    match_url: str | None
    event_name: str | None
    round_name: str | None
    court_name: str | None
    status_text: str | None
    status_code: str | None
    home_name: str | None
    away_name: str | None
    winner_side: str | None
    score_text: str | None
    point_score_home: str | None
    point_score_away: str | None
    discovery_source: str | None
    schedule_payload: dict[str, Any]
    detail_payload: dict[str, Any]
    stats_payload: dict[str, Any]
    history_payload: dict[str, Any]
    keys_payload: dict[str, Any]
    insights_payload: dict[str, Any]
    page_payload: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "slam_code": self.slam_code,
            "provider_name": self.provider_name,
            "source_match_id": self.source_match_id,
            "season_year": self.season_year,
            "match_url": self.match_url,
            "event_name": self.event_name,
            "round_name": self.round_name,
            "court_name": self.court_name,
            "status_text": self.status_text,
            "status_code": self.status_code,
            "home_name": self.home_name,
            "away_name": self.away_name,
            "winner_side": self.winner_side,
            "score_text": self.score_text,
            "point_score_home": self.point_score_home,
            "point_score_away": self.point_score_away,
            "discovery_source": self.discovery_source,
            "schedule_payload": self.schedule_payload,
            "detail_payload": self.detail_payload,
            "stats_payload": self.stats_payload,
            "history_payload": self.history_payload,
            "keys_payload": self.keys_payload,
            "insights_payload": self.insights_payload,
            "page_payload": self.page_payload,
        }


@dataclass
class PointEventRow:
    match_id: str
    set_no: int
    game_no: int
    point_no: int
    server_id: str | None
    returner_id: str | None
    score_state: str | None
    break_point_flag: bool
    ace_flag: bool
    winner_flag: bool
    unforced_error_flag: bool
    rally_count: int | None
    serve_speed: int | None
    serve_direction: str | None
    return_depth: str | None
    point_winner_id: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "set_no": self.set_no,
            "game_no": self.game_no,
            "point_no": self.point_no,
            "server_id": self.server_id,
            "returner_id": self.returner_id,
            "score_state": self.score_state,
            "break_point_flag": self.break_point_flag,
            "ace_flag": self.ace_flag,
            "winner_flag": self.winner_flag,
            "unforced_error_flag": self.unforced_error_flag,
            "rally_count": self.rally_count,
            "serve_speed": self.serve_speed,
            "serve_direction": self.serve_direction,
            "return_depth": self.return_depth,
            "point_winner_id": self.point_winner_id,
        }


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _clean_int(value: Any) -> int | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _build_session(session: requests.Session | None = None) -> requests.Session:
    if session is not None:
        session.trust_env = False
        if "User-Agent" not in session.headers:
            session.headers["User-Agent"] = DEFAULT_USER_AGENT
        return session
    built = requests.Session()
    built.trust_env = False
    built.headers["User-Agent"] = DEFAULT_USER_AGENT
    return built


def _json_or_error(response: requests.Response) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": response.url,
        "status_code": response.status_code,
    }
    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower():
        try:
            payload["data"] = response.json()
            return payload
        except ValueError:
            pass
    try:
        payload["data"] = response.json()
    except ValueError:
        payload["text"] = response.text[:5000]
    return payload


def _fetch_json(
    url: str,
    *,
    referer: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    active_session = _build_session(session)
    resolved_url = urljoin(referer, url) if referer else url
    headers = {"Referer": referer} if referer else None
    response = active_session.get(resolved_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_optional_json_payload(
    url: str | None,
    *,
    referer: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    if not url:
        return {}
    active_session = _build_session(session)
    resolved_url = urljoin(referer, url) if referer else url
    headers = {"Referer": referer} if referer else None
    response = active_session.get(resolved_url, headers=headers, timeout=30)
    return _json_or_error(response)


def _fetch_page_metadata(
    url: str,
    *,
    referer: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    active_session = _build_session(session)
    resolved_url = urljoin(referer, url) if referer else url
    headers = {"Referer": referer} if referer else None
    response = active_session.get(resolved_url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    meta = {
        "url": response.url,
        "status_code": response.status_code,
        "title": _clean_text(soup.title.get_text(" ", strip=True) if soup.title else None),
        "canonical_url": None,
        "description": None,
        "og_title": None,
        "og_description": None,
    }
    canonical = soup.find("link", rel="canonical")
    if canonical is not None:
        meta["canonical_url"] = _clean_text(canonical.get("href"))
    for tag in soup.find_all("meta"):
        name = _clean_text(tag.get("name") or tag.get("property"))
        content = _clean_text(tag.get("content"))
        if name == "description":
            meta["description"] = content
        elif name == "og:title":
            meta["og_title"] = content
        elif name == "og:description":
            meta["og_description"] = content
    return meta


def _stringify_point_score(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if cleaned in {None, "", "0"}:
        return None
    return cleaned


def _is_truthy_flag(value: Any) -> bool:
    cleaned = _clean_text(value)
    return cleaned not in {None, "", "0", "False", "false"}


def _is_numeric_point_number(value: Any) -> bool:
    cleaned = _clean_text(value)
    return bool(cleaned and cleaned.isdigit())


def _competitor_player_ids(team_payload: dict[str, Any]) -> list[str]:
    ids = [
        _clean_text(team_payload.get("idA")),
        _clean_text(team_payload.get("idB")),
    ]
    return [player_id for player_id in ids if player_id]


def _single_player_id(team_payload: dict[str, Any]) -> str | None:
    ids = _competitor_player_ids(team_payload)
    return ids[0] if len(ids) == 1 else None


def _snapshot_side_player_ids(snapshot: GrandSlamMatchSnapshot) -> dict[str, str | None]:
    detail_payload = snapshot.detail_payload.get("data")
    if not isinstance(detail_payload, dict):
        detail_payload = snapshot.schedule_payload
    if not isinstance(detail_payload, dict):
        return {"1": None, "2": None}
    team_1 = detail_payload.get("team1") or {}
    team_2 = detail_payload.get("team2") or {}
    if not isinstance(team_1, dict):
        team_1 = {}
    if not isinstance(team_2, dict):
        team_2 = {}
    return {
        "1": _single_player_id(team_1),
        "2": _single_player_id(team_2),
    }


def _score_state_from_history_row(row: dict[str, Any]) -> str | None:
    left = _clean_text(row.get("P1Score"))
    right = _clean_text(row.get("P2Score"))
    if left is None and right is None:
        return None
    return f"{left or ''}-{right or ''}".strip("-")


def _history_rows(snapshot: GrandSlamMatchSnapshot) -> list[dict[str, Any]]:
    rows = snapshot.history_payload.get("data")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def extract_point_events_from_snapshot(
    snapshot: GrandSlamMatchSnapshot,
    *,
    match_id: str,
) -> list[PointEventRow]:
    side_player_ids = _snapshot_side_player_ids(snapshot)
    point_events: list[PointEventRow] = []
    for row in _history_rows(snapshot):
        if not _is_numeric_point_number(row.get("PointNumber")):
            continue
        point_winner_side = _clean_text(row.get("PointWinner"))
        server_side = _clean_text(row.get("PointServer"))
        if point_winner_side not in {"1", "2"}:
            continue
        if server_side not in {"1", "2"}:
            server_side = None
        returner_side = None
        if server_side == "1":
            returner_side = "2"
        elif server_side == "2":
            returner_side = "1"

        set_no = _clean_int(row.get("SetNo"))
        game_no = _clean_int(row.get("GameNo"))
        point_no = _clean_int(row.get("PointNumber"))
        if set_no is None or game_no is None or point_no is None:
            continue

        break_point_flag = _is_truthy_flag(row.get("BreakPoint")) or _is_truthy_flag(
            row.get("BreakPointOpportunity")
        )
        point_events.append(
            PointEventRow(
                match_id=match_id,
                set_no=set_no,
                game_no=game_no,
                point_no=point_no,
                server_id=side_player_ids.get(server_side or ""),
                returner_id=side_player_ids.get(returner_side or ""),
                score_state=_score_state_from_history_row(row),
                break_point_flag=break_point_flag,
                ace_flag=_is_truthy_flag(row.get("Ace")),
                winner_flag=_is_truthy_flag(row.get("Winner")),
                unforced_error_flag=_is_truthy_flag(row.get("UnforcedError")),
                rally_count=_clean_int(row.get("RallyCount")),
                serve_speed=_clean_int(row.get("Speed_KMH")),
                serve_direction=_clean_text(row.get("ServeWidth")),
                return_depth=_clean_text(row.get("ReturnDepth")),
                point_winner_id=side_player_ids.get(point_winner_side),
            )
        )
    return point_events


def snapshot_has_point_history(snapshot: GrandSlamMatchSnapshot) -> bool:
    return bool(_history_rows(snapshot))


def _wimbledon_like_player_name(team: dict[str, Any], prefix: str) -> str | None:
    names = [
        _clean_text(team.get(f"displayName{prefix}")),
        _clean_text(
            " ".join(
                part
                for part in [team.get(f"firstName{prefix}"), team.get(f"lastName{prefix}")]
                if _clean_text(part)
            )
        ),
    ]
    for name in names:
        if name:
            return name
    return None


def _wimbledon_like_team_display(team: dict[str, Any]) -> str | None:
    players = [
        name
        for name in [
            _wimbledon_like_player_name(team, "A"),
            _wimbledon_like_player_name(team, "B"),
        ]
        if name
    ]
    if not players:
        return None
    return " / ".join(players)


def _wimbledon_like_score_text(match_payload: dict[str, Any]) -> str | None:
    score = _clean_text(match_payload.get("score"))
    if score:
        return score
    sets = match_payload.get("sets")
    if isinstance(sets, list):
        parts: list[str] = []
        for set_payload in sets:
            if not isinstance(set_payload, dict):
                continue
            team_1 = _clean_text(set_payload.get("team1")) or _clean_text(set_payload.get("player1"))
            team_2 = _clean_text(set_payload.get("team2")) or _clean_text(set_payload.get("player2"))
            if team_1 is None and team_2 is None:
                continue
            parts.append(f"{team_1 or ''}-{team_2 or ''}".strip("-"))
        if parts:
            return " ".join(parts)
    return None


def _extract_season_year_from_url(url: str | None) -> int | None:
    if not url:
        return None
    match = re.search(r"/(20\d{2})/", url)
    return int(match.group(1)) if match else None


def _expand_url(template: str | None, match_id: str) -> str | None:
    if not template:
        return None
    return template.replace("<matchId>", match_id).replace("<match_id>", match_id)


def _replace_season_year_in_url(url: str | None, season_year: int | None) -> str | None:
    cleaned = _clean_text(url)
    if cleaned is None or season_year is None:
        return cleaned
    return re.sub(r"/20\d{2}/", f"/{season_year}/", cleaned, count=1)


def _wimbledon_like_snapshots(
    *,
    slam_code: str,
    provider_name: str,
    config_url: str,
    referer: str,
    season_year: int | None = None,
    session: requests.Session | None = None,
) -> list[GrandSlamMatchSnapshot]:
    logger = logging.getLogger("grand_slam_pipeline")
    logger.info(
        "Discovering %s snapshots | season_year=%s",
        slam_code,
        season_year or "current",
    )
    config = _fetch_json(config_url, referer=referer, session=session)
    scoring = config["scoringData"]
    other_data = config.get("otherData", {})
    innovations = other_data.get("innovations", {})

    live_score_url = _replace_season_year_in_url(scoring["liveScore"]["path"], season_year)
    completed_days_url = _replace_season_year_in_url(scoring["completedMatchDays"], season_year)
    effective_season_year = season_year or _extract_season_year_from_url(completed_days_url)

    live_payload: dict[str, Any] = {"matches": []}
    if effective_season_year is None or effective_season_year == _extract_season_year_from_url(scoring["liveScore"]["path"]):
        live_payload = _fetch_json(live_score_url, referer=referer, session=session)
    completed_days_payload = _fetch_json(completed_days_url, referer=referer, session=session)

    discovered_matches: dict[str, tuple[dict[str, Any], str]] = {}
    for match_payload in live_payload.get("matches", []):
        match_id = _clean_text(match_payload.get("match_id"))
        if match_id:
            discovered_matches[match_id] = (match_payload, "live_scores")

    for event_day in completed_days_payload.get("eventDays", []):
        day_url = _replace_season_year_in_url(
            _clean_text(event_day.get("url")) or _clean_text(event_day.get("feedUrl")),
            effective_season_year,
        )
        if not day_url:
            continue
        day_payload = _fetch_json(day_url, referer=referer, session=session)
        discovery_source = f"completed_day_{event_day.get('tournDay')}"
        for match_payload in day_payload.get("matches", []):
            match_id = _clean_text(match_payload.get("match_id"))
            if match_id:
                discovered_matches[match_id] = (match_payload, discovery_source)

    logger.info(
        "Discovered %s %s matches | season_year=%s",
        len(discovered_matches),
        slam_code,
        effective_season_year or "current",
    )
    snapshots: list[GrandSlamMatchSnapshot] = []
    total_matches = len(discovered_matches)
    for index, (match_id, (schedule_match, discovery_source)) in enumerate(discovered_matches.items(), start=1):
        if index == 1 or index % 25 == 0 or index == total_matches:
            logger.info(
                "Fetching %s match payloads | season_year=%s progress=%s/%s match_id=%s",
                slam_code,
                effective_season_year or "current",
                index,
                total_matches,
                match_id,
            )
        detail_payload = _fetch_optional_json_payload(
            _expand_url(
                _replace_season_year_in_url(_clean_text(scoring.get("completedMatch")), effective_season_year),
                match_id,
            ),
            referer=referer,
            session=session,
        )
        stats_payload = _fetch_optional_json_payload(
            _expand_url(
                _replace_season_year_in_url(_clean_text(scoring.get("matchStatistics")), effective_season_year),
                match_id,
            ),
            referer=referer,
            session=session,
        )
        history_path = _replace_season_year_in_url(
            scoring.get("matchHistory", {}).get("path"),
            effective_season_year,
        )
        history_payload = _fetch_optional_json_payload(
            _expand_url(_clean_text(history_path), match_id),
            referer=referer,
            session=session,
        )
        keys_path = _replace_season_year_in_url(scoring.get("keys", {}).get("path"), effective_season_year)
        keys_payload = _fetch_optional_json_payload(
            _expand_url(_clean_text(keys_path), match_id),
            referer=referer,
            session=session,
        )
        insight_template = (
            _replace_season_year_in_url(_clean_text(scoring.get("matchInsights")), effective_season_year)
            or _replace_season_year_in_url(_clean_text(innovations.get("matchInsights")), effective_season_year)
            or _replace_season_year_in_url(
                _clean_text(innovations.get("matchInsightsFacts")),
                effective_season_year,
            )
        )
        insights_payload = _fetch_optional_json_payload(
            _expand_url(insight_template, match_id),
            referer=referer,
            session=session,
        )

        detail_data = detail_payload.get("data")
        effective_match = detail_data if isinstance(detail_data, dict) else schedule_match
        team_1 = effective_match.get("team1") or {}
        team_2 = effective_match.get("team2") or {}
        snapshots.append(
            GrandSlamMatchSnapshot(
                slam_code=slam_code,
                provider_name=provider_name,
                source_match_id=match_id,
                season_year=effective_season_year,
                match_url=None,
                event_name=_clean_text(effective_match.get("eventName")),
                round_name=_clean_text(effective_match.get("roundName")),
                court_name=_clean_text(effective_match.get("courtName")),
                status_text=_clean_text(effective_match.get("status")),
                status_code=_clean_text(effective_match.get("statusCode")),
                home_name=_wimbledon_like_team_display(team_1),
                away_name=_wimbledon_like_team_display(team_2),
                winner_side=_clean_text(effective_match.get("winner")),
                score_text=_wimbledon_like_score_text(effective_match),
                point_score_home=_stringify_point_score(effective_match.get("team1Point")),
                point_score_away=_stringify_point_score(effective_match.get("team2Point")),
                discovery_source=discovery_source,
                schedule_payload=schedule_match,
                detail_payload=detail_payload,
                stats_payload=stats_payload,
                history_payload=history_payload,
                keys_payload=keys_payload,
                insights_payload=insights_payload,
                page_payload={},
            )
        )
    return snapshots


def _decode_embedded_html(text: str) -> str:
    decoded = text
    for _ in range(3):
        decoded = html.unescape(decoded)
    return (
        decoded.replace("\\/", "/")
        .replace("\\u002F", "/")
        .replace("\\u003C", "<")
        .replace("\\u003E", ">")
        .replace("\\u0026", "&")
    )


def _ao_lookup_by_uuid(items: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("uuid")): item
        for item in items
        if isinstance(item, dict) and _clean_text(item.get("uuid"))
    }


def _ao_player_display_name(player: dict[str, Any]) -> str | None:
    return _clean_text(player.get("full_name")) or _clean_text(
        " ".join(part for part in [player.get("first_name"), player.get("last_name")] if _clean_text(part))
    )


def _ao_expand_team(team_ref: dict[str, Any], teams_by_uuid: dict[str, dict[str, Any]], players_by_uuid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    team_id = _clean_text(team_ref.get("team_id"))
    team_payload = teams_by_uuid.get(team_id or "", {})
    players = [
        players_by_uuid[player_uuid]
        for player_uuid in team_payload.get("players", [])
        if player_uuid in players_by_uuid
    ]
    expanded = dict(team_ref)
    expanded["team"] = team_payload
    expanded["players"] = players
    return expanded


def _ao_team_display(team_payload: dict[str, Any]) -> str | None:
    players = [_ao_player_display_name(player) for player in team_payload.get("players", []) if isinstance(player, dict)]
    players = [player for player in players if player]
    if not players:
        return None
    return " / ".join(players)


def _ao_score_text(match: dict[str, Any]) -> str | None:
    teams = match.get("teams")
    if not isinstance(teams, list) or len(teams) < 2:
        return None
    left_scores = teams[0].get("score") or []
    right_scores = teams[1].get("score") or []
    if not isinstance(left_scores, list) or not isinstance(right_scores, list):
        return None
    parts: list[str] = []
    for left_set, right_set in zip(left_scores, right_scores):
        if not isinstance(left_set, dict) or not isinstance(right_set, dict):
            continue
        left_game = _clean_text(left_set.get("game"))
        right_game = _clean_text(right_set.get("game"))
        if left_game is None and right_game is None:
            continue
        parts.append(f"{left_game or ''}-{right_game or ''}".strip("-"))
    return " ".join(parts) or None


def _ao_winner_side(expanded_teams: Sequence[dict[str, Any]]) -> str | None:
    for index, team in enumerate(expanded_teams[:2], start=1):
        if _clean_text(team.get("status")) == "Winner":
            return str(index)
    return None


def _australian_open_snapshots(
    *,
    include_match_pages: bool = False,
    session: requests.Session | None = None,
) -> list[GrandSlamMatchSnapshot]:
    active_session = _build_session(session)
    logger = logging.getLogger("grand_slam_pipeline")
    referer = AO_RESULTS_URL
    response = active_session.get(AO_RESULTS_URL, headers={"Referer": AO_RESULTS_URL}, timeout=30)
    response.raise_for_status()
    decoded_html = _decode_embedded_html(response.text)
    result_urls = sorted(
        dict.fromkeys(
            re.findall(
                r"https://prod-scores-api\.ausopen\.com/year/\d+/period/[A-Z]+/day/\d+/results",
                decoded_html,
            )
        )
    )
    snapshots: list[GrandSlamMatchSnapshot] = []
    for result_url in result_urls:
        logger.info("Fetching australian_open results page | url=%s", result_url)
        result_payload = _fetch_json(result_url, referer=referer, session=active_session)
        teams_by_uuid = _ao_lookup_by_uuid(result_payload.get("teams", []))
        players_by_uuid = _ao_lookup_by_uuid(result_payload.get("players", []))
        events_by_uuid = _ao_lookup_by_uuid(result_payload.get("events", []))
        courts_by_uuid = _ao_lookup_by_uuid(result_payload.get("courts", []))
        year_payload = result_payload.get("year") or {}
        season_year = None
        try:
            season_year = int(year_payload.get("year"))
        except (TypeError, ValueError):
            season_year = _extract_season_year_from_url(result_url)

        for match in result_payload.get("matches", []):
            if not isinstance(match, dict):
                continue
            match_id = _clean_text(match.get("match_id")) or _clean_text(match.get("uuid"))
            if not match_id:
                continue
            expanded_teams = [
                _ao_expand_team(team_ref, teams_by_uuid, players_by_uuid)
                for team_ref in match.get("teams", [])
                if isinstance(team_ref, dict)
            ]
            event_payload = events_by_uuid.get(_clean_text(match.get("event_uuid")) or "", {})
            court_payload = courts_by_uuid.get(_clean_text(match.get("court_id")) or "", {})
            detail_payload = {
                "match": match,
                "tournament": result_payload.get("tournament"),
                "year": result_payload.get("year"),
                "event": event_payload,
                "court": court_payload,
                "teams": expanded_teams,
            }
            match_url = _clean_text(match.get("match_centre_link"))
            page_payload = {}
            if include_match_pages and match_url:
                page_payload = _fetch_page_metadata(match_url, referer=referer, session=active_session)
            snapshots.append(
                GrandSlamMatchSnapshot(
                    slam_code="australian_open",
                    provider_name="australian_open_official",
                    source_match_id=match_id,
                    season_year=season_year,
                    match_url=match_url,
                    event_name=_clean_text(event_payload.get("name")) or _clean_text(event_payload.get("title")),
                    round_name=_clean_text(match.get("match_state")) or _clean_text(match.get("round")),
                    court_name=_clean_text(court_payload.get("name")),
                    status_text=_clean_text((match.get("match_status") or {}).get("name")) or _clean_text(match.get("match_state")),
                    status_code=_clean_text((match.get("match_status") or {}).get("code")) or _clean_text((match.get("match_status") or {}).get("abbr")),
                    home_name=_ao_team_display(expanded_teams[0]) if len(expanded_teams) > 0 else None,
                    away_name=_ao_team_display(expanded_teams[1]) if len(expanded_teams) > 1 else None,
                    winner_side=_ao_winner_side(expanded_teams),
                    score_text=_ao_score_text({"teams": expanded_teams}),
                    point_score_home=None,
                    point_score_away=None,
                    discovery_source=result_url,
                    schedule_payload={"results_url": result_url},
                    detail_payload=detail_payload,
                    stats_payload={},
                    history_payload={},
                    keys_payload={},
                    insights_payload={},
                    page_payload=page_payload,
                )
            )
    return snapshots


def _rg_extract_match_urls(results_page_html: str) -> list[str]:
    decoded = _decode_embedded_html(results_page_html)
    urls = sorted(set(re.findall(r"/en-us/matches/\d{4}/[A-Z0-9]+", decoded)))
    return [urljoin(RG_BASE_URL, path) for path in urls]


def _roland_garros_snapshots(
    *,
    event_codes: Sequence[str] = DEFAULT_RG_EVENT_CODES,
    include_match_pages: bool = True,
    session: requests.Session | None = None,
) -> list[GrandSlamMatchSnapshot]:
    active_session = _build_session(session)
    logger = logging.getLogger("grand_slam_pipeline")
    snapshots: list[GrandSlamMatchSnapshot] = []
    for event_code in event_codes:
        results_url = RG_RESULTS_URL_TEMPLATE.format(event_code=event_code)
        logger.info("Fetching roland_garros results page | event_code=%s", event_code)
        response = active_session.get(results_url, headers={"Referer": results_url}, timeout=30)
        response.raise_for_status()
        match_urls = _rg_extract_match_urls(response.text)
        results_page_payload = _fetch_page_metadata(results_url, referer=results_url, session=active_session)
        logger.info(
            "Discovered %s roland_garros match pages | event_code=%s",
            len(match_urls),
            event_code,
        )
        for match_url in match_urls:
            page_payload = {}
            if include_match_pages:
                page_payload = _fetch_page_metadata(match_url, referer=results_url, session=active_session)
            season_year = _extract_season_year_from_url(match_url)
            source_match_id = _clean_text(match_url.rstrip("/").rsplit("/", 1)[-1]) or match_url
            title = _clean_text(page_payload.get("og_title")) or _clean_text(page_payload.get("title"))
            home_name = None
            away_name = None
            if title and " vs " in title:
                home_name, away_name = [_clean_text(part) for part in title.split(" vs ", 1)]
            snapshots.append(
                GrandSlamMatchSnapshot(
                    slam_code="roland_garros",
                    provider_name="roland_garros_official",
                    source_match_id=source_match_id,
                    season_year=season_year,
                    match_url=match_url,
                    event_name=event_code,
                    round_name=None,
                    court_name=None,
                    status_text=None,
                    status_code=None,
                    home_name=home_name,
                    away_name=away_name,
                    winner_side=None,
                    score_text=None,
                    point_score_home=None,
                    point_score_away=None,
                    discovery_source=results_url,
                    schedule_payload={"results_page": results_page_payload},
                    detail_payload={"event_code": event_code, "results_url": results_url},
                    stats_payload={},
                    history_payload={},
                    keys_payload={},
                    insights_payload={},
                    page_payload=page_payload,
                )
            )
    return snapshots


def fetch_grand_slam_match_snapshots(
    *,
    slams: Sequence[str] = DEFAULT_SLAMS,
    include_match_pages: bool = False,
    rg_event_codes: Sequence[str] = DEFAULT_RG_EVENT_CODES,
    season_years: Sequence[int] | None = None,
    session: requests.Session | None = None,
) -> list[GrandSlamMatchSnapshot]:
    resolved_slams = [slam for slam in dict.fromkeys(slams) if _clean_text(slam)]
    resolved_season_years = [season_year for season_year in dict.fromkeys(season_years or []) if season_year]
    active_session = _build_session(session)
    logger = logging.getLogger("grand_slam_pipeline")
    snapshots: list[GrandSlamMatchSnapshot] = []
    for slam in resolved_slams:
        logger.info(
            "Starting slam discovery | slam=%s season_years=%s",
            slam,
            resolved_season_years or ["current"],
        )
        if slam == "wimbledon":
            target_years = resolved_season_years or [None]
            for season_year in target_years:
                snapshots.extend(
                    _wimbledon_like_snapshots(
                        slam_code="wimbledon",
                        provider_name="wimbledon_official",
                        config_url=WIMBLEDON_CONFIG_URL,
                        referer="https://www.wimbledon.com/en_GB/scores/index.html",
                        season_year=season_year,
                        session=active_session,
                    )
                )
        elif slam == "us_open":
            target_years = resolved_season_years or [None]
            for season_year in target_years:
                snapshots.extend(
                    _wimbledon_like_snapshots(
                        slam_code="us_open",
                        provider_name="us_open_official",
                        config_url=US_OPEN_CONFIG_URL,
                        referer="https://www.usopen.org/en_US/scores/index.html",
                        season_year=season_year,
                        session=active_session,
                    )
                )
        elif slam == "australian_open":
            snapshots.extend(
                _australian_open_snapshots(
                    include_match_pages=include_match_pages,
                    session=active_session,
                )
            )
        elif slam == "roland_garros":
            snapshots.extend(
                _roland_garros_snapshots(
                    event_codes=rg_event_codes,
                    include_match_pages=include_match_pages,
                    session=active_session,
                )
            )
        else:
            raise ValueError(f"Unsupported Grand Slam code: {slam}")
    return snapshots
