from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


FLASHSCORE_BASE_URL = "https://www.flashscore.com"
DEFAULT_TENNIS_INDEX_URL = f"{FLASHSCORE_BASE_URL}/tennis/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ENVIRONMENT_RE = re.compile(r"window\.environment\s*=\s*(\{.*?\});\s*</script>", re.S)
MATCH_URL_RE = re.compile(r"/match/tennis/[^\"']+/")
TOURNAMENT_URL_RE = re.compile(r"^/tennis/(?P<tour>atp-singles|challenger-men-singles)/[^/]+/$")
CURRENT_TOURNAMENTS_MENU_ID = "mt"


@dataclass
class FlashscoreMatchSnapshot:
    match_url: str
    event_id: str
    project_type_id: int
    feed_sign: str
    home_name: str | None
    away_name: str | None
    home_participant_id: str | None
    away_participant_id: str | None
    tournament_name: str | None
    category_name: str | None
    country_name: str | None
    point_score_home: str | None
    point_score_away: str | None
    current_game_feed_raw: str
    current_game_feed: dict[str, Any]
    common_feed_raw: str
    common_feed: dict[str, Any]
    match_history_feed_raw: str
    game_feed_raw: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "match_url": self.match_url,
            "event_id": self.event_id,
            "project_type_id": self.project_type_id,
            "feed_sign": self.feed_sign,
            "home_name": self.home_name,
            "away_name": self.away_name,
            "home_participant_id": self.home_participant_id,
            "away_participant_id": self.away_participant_id,
            "tournament_name": self.tournament_name,
            "category_name": self.category_name,
            "country_name": self.country_name,
            "point_score_home": self.point_score_home,
            "point_score_away": self.point_score_away,
            "current_game_feed_raw": self.current_game_feed_raw,
            "current_game_feed": self.current_game_feed,
            "common_feed_raw": self.common_feed_raw,
            "common_feed": self.common_feed,
            "match_history_feed_raw": self.match_history_feed_raw,
            "game_feed_raw": self.game_feed_raw,
        }


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _normalize_feed_value(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if cleaned in {None, "", "EMPTY"}:
        return None
    return cleaned


def _build_session(session: requests.Session | None = None) -> requests.Session:
    if session is not None:
        if "User-Agent" not in session.headers:
            session.headers["User-Agent"] = DEFAULT_USER_AGENT
        return session
    built = requests.Session()
    built.headers["User-Agent"] = DEFAULT_USER_AGENT
    return built


def parse_compact_feed_records(feed_blob: str) -> list[dict[str, str | None]]:
    if not feed_blob:
        return []
    records: list[dict[str, str | None]] = []
    for raw_record in feed_blob.split("~"):
        raw_record = raw_record.strip()
        if not raw_record:
            continue
        parsed_record: dict[str, str | None] = {}
        for field in raw_record.split("¬"):
            field = field.strip()
            if not field:
                continue
            key, separator, value = field.partition("÷")
            key = key.strip()
            if not key:
                continue
            parsed_record[key] = value if separator else None
        if parsed_record:
            records.append(parsed_record)
    return records


def flatten_compact_feed(feed_blob: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for record in parse_compact_feed_records(feed_blob):
        for key, value in record.items():
            if key == "A1":
                continue
            if key not in flattened:
                flattened[key] = value
                continue
            existing = flattened[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                flattened[key] = [existing, value]
    return flattened


def extract_environment_from_html(html: str) -> dict[str, Any]:
    match = ENVIRONMENT_RE.search(html)
    if match is None:
        raise ValueError("Flashscore page did not expose window.environment")
    return json.loads(match.group(1))


def discover_tennis_match_urls(
    index_url: str = DEFAULT_TENNIS_INDEX_URL,
    *,
    session: requests.Session | None = None,
) -> list[str]:
    active_session = _build_session(session)
    response = active_session.get(index_url, timeout=30)
    response.raise_for_status()
    matches = MATCH_URL_RE.findall(response.text)
    discovered: list[str] = []
    seen: set[str] = set()
    for match_path in matches:
        match_url = urljoin(index_url, match_path)
        if match_url not in seen:
            seen.add(match_url)
            discovered.append(match_url)
    return discovered


def discover_current_tournament_urls(
    index_url: str = DEFAULT_TENNIS_INDEX_URL,
    *,
    include_tours: tuple[str, ...] = ("atp-singles", "challenger-men-singles"),
    session: requests.Session | None = None,
) -> list[str]:
    active_session = _build_session(session)
    response = active_session.get(index_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    current_menu = soup.find(id=CURRENT_TOURNAMENTS_MENU_ID)
    search_root = current_menu if current_menu is not None else soup

    discovered: list[str] = []
    seen: set[str] = set()
    for anchor in search_root.find_all("a", href=True):
        href = anchor["href"]
        match = TOURNAMENT_URL_RE.match(href)
        if match is None or match.group("tour") not in include_tours:
            continue
        tournament_url = urljoin(index_url, href)
        if tournament_url in seen:
            continue
        seen.add(tournament_url)
        discovered.append(tournament_url)
    return discovered


def discover_match_urls_from_tournament_page(
    tournament_url: str,
    *,
    include_tabs: tuple[str, ...] = ("summary", "fixtures"),
    session: requests.Session | None = None,
) -> list[str]:
    active_session = _build_session(session)
    discovered: list[str] = []
    seen: set[str] = set()

    for tab in include_tabs:
        page_url = tournament_url if tab == "summary" else urljoin(tournament_url, f"{tab}/")
        response = active_session.get(page_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        extra_content = soup.select_one("aside#extraContent")
        if extra_content is None:
            continue
        for anchor in extra_content.select('a[href*="/match/tennis/"]'):
            match_url = urljoin(page_url, anchor.get("href", ""))
            if match_url in seen:
                continue
            seen.add(match_url)
            discovered.append(match_url)
    return discovered


def discover_current_atp_challenger_match_urls(
    index_url: str = DEFAULT_TENNIS_INDEX_URL,
    *,
    include_tours: tuple[str, ...] = ("atp-singles", "challenger-men-singles"),
    include_tabs: tuple[str, ...] = ("summary", "fixtures"),
    session: requests.Session | None = None,
) -> list[str]:
    active_session = _build_session(session)
    tournament_urls = discover_current_tournament_urls(
        index_url,
        include_tours=include_tours,
        session=active_session,
    )
    discovered: list[str] = []
    seen: set[str] = set()
    for tournament_url in tournament_urls:
        for match_url in discover_match_urls_from_tournament_page(
            tournament_url,
            include_tabs=include_tabs,
            session=active_session,
        ):
            if match_url in seen:
                continue
            seen.add(match_url)
            discovered.append(match_url)
    return discovered


def fetch_match_environment(
    match_url: str,
    *,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], str]:
    active_session = _build_session(session)
    response = active_session.get(match_url, timeout=30)
    response.raise_for_status()
    return extract_environment_from_html(response.text), response.url


def fetch_feed_blob(
    event_id: str,
    *,
    project_type_id: int,
    feed_prefix: str,
    feed_sign: str,
    session: requests.Session | None = None,
) -> str:
    active_session = _build_session(session)
    feed_name = f"{feed_prefix}_{project_type_id}_{event_id}"
    response = active_session.get(
        f"{FLASHSCORE_BASE_URL}/x/feed/{feed_name}",
        headers={"x-fsign": feed_sign},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def fetch_match_snapshot(
    match_url: str,
    *,
    session: requests.Session | None = None,
) -> FlashscoreMatchSnapshot:
    active_session = _build_session(session)
    environment, resolved_url = fetch_match_environment(match_url, session=active_session)

    config_app = environment["config"]["app"]
    project_type_id = int(config_app["project_type"]["id"])
    feed_sign = str(config_app["feed_sign"])
    event_id = str(environment["event_id_c"])

    participants = environment.get("participantsData") or {}
    home = (participants.get("home") or [{}])[0]
    away = (participants.get("away") or [{}])[0]
    header = environment.get("header") or {}
    tournament = header.get("tournament") or {}

    current_game_feed_raw = fetch_feed_blob(
        event_id,
        project_type_id=project_type_id,
        feed_prefix="df_mhs",
        feed_sign=feed_sign,
        session=active_session,
    )
    common_feed_raw = fetch_feed_blob(
        event_id,
        project_type_id=project_type_id,
        feed_prefix="dc",
        feed_sign=feed_sign,
        session=active_session,
    )
    match_history_feed_raw = fetch_feed_blob(
        event_id,
        project_type_id=project_type_id,
        feed_prefix="df_mh",
        feed_sign=feed_sign,
        session=active_session,
    )
    game_feed_raw = fetch_feed_blob(
        event_id,
        project_type_id=project_type_id,
        feed_prefix="g",
        feed_sign=feed_sign,
        session=active_session,
    )

    current_game_feed = flatten_compact_feed(current_game_feed_raw)
    common_feed = flatten_compact_feed(common_feed_raw)

    return FlashscoreMatchSnapshot(
        match_url=resolved_url,
        event_id=event_id,
        project_type_id=project_type_id,
        feed_sign=feed_sign,
        home_name=_clean_text(home.get("seo_name") or home.get("name")),
        away_name=_clean_text(away.get("seo_name") or away.get("name")),
        home_participant_id=_clean_text(home.get("id")),
        away_participant_id=_clean_text(away.get("id")),
        tournament_name=_clean_text(tournament.get("tournament")),
        category_name=_clean_text(tournament.get("category")),
        country_name=_clean_text(header.get("country_name")),
        point_score_home=_normalize_feed_value(current_game_feed.get("TS")),
        point_score_away=_normalize_feed_value(current_game_feed.get("TE")),
        current_game_feed_raw=current_game_feed_raw,
        current_game_feed=current_game_feed,
        common_feed_raw=common_feed_raw,
        common_feed=common_feed,
        match_history_feed_raw=match_history_feed_raw,
        game_feed_raw=game_feed_raw,
    )
