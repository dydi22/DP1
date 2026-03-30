from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import requests

from tennis_model.names import normalize_player_name


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
PUBLIC_TOP_URL = "https://api.universaltennis.com/v3/player/top"
PUBLIC_SEARCH_URL = "https://api.utrsports.net/v2/search"
DEFAULT_TOP_COUNT = 200
DEFAULT_SEARCH_TOP = 10
DEFAULT_WORKERS = 8
CSV_COLUMNS = [
    "player_name",
    "rating_date",
    "utr_singles",
    "utr_rank",
    "three_month_rating",
    "nationality",
    "provider_player_id",
]


@dataclass
class UtrPublicScrapeSummary:
    tracked_players: int = 0
    top_rankings_rows: int = 0
    top_rankings_matches: int = 0
    search_matches: int = 0
    unmatched_players: int = 0
    rows_written: int = 0
    output_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracked_players": self.tracked_players,
            "top_rankings_rows": self.top_rankings_rows,
            "top_rankings_matches": self.top_rankings_matches,
            "search_matches": self.search_matches,
            "unmatched_players": self.unmatched_players,
            "rows_written": self.rows_written,
            "output_csv": self.output_csv,
        }


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _clean_float(value: Any) -> float | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


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
        if "User-Agent" not in session.headers:
            session.headers["User-Agent"] = DEFAULT_USER_AGENT
        session.headers.setdefault("Accept", "application/json")
        return session
    built = requests.Session()
    built.headers["User-Agent"] = DEFAULT_USER_AGENT
    built.headers["Accept"] = "application/json"
    return built


def fetch_public_top_players(
    *,
    gender: str = "m",
    tags: str = "Pro",
    count: int = DEFAULT_TOP_COUNT,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    active_session = _build_session(session)
    response = active_session.get(
        PUBLIC_TOP_URL,
        params={
            "gender": gender,
            "tags": tags,
            "count": count,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def fetch_public_player_search_hits(
    query: str,
    *,
    top: int = DEFAULT_SEARCH_TOP,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    active_session = _build_session(session)
    response = active_session.get(
        PUBLIC_SEARCH_URL,
        params={
            "query": query,
            "top": top,
            "skip": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    players_payload = payload.get("players") if isinstance(payload, dict) else None
    hits = players_payload.get("hits") if isinstance(players_payload, dict) else None
    if not isinstance(hits, list):
        return []
    results: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get("source")
        if isinstance(source, dict):
            results.append(source)
    return results


def _exact_name_matches(player_name: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = normalize_player_name(player_name)
    return [
        candidate
        for candidate in candidates
        if normalize_player_name(_clean_text(candidate.get("displayName")) or "") == lookup
    ]


def _score_search_candidate(candidate: dict[str, Any], *, ioc: str | None) -> tuple[int, float]:
    score = 0
    nationality = _clean_text(candidate.get("nationality"))
    if ioc and nationality == ioc:
        score += 100
    if candidate.get("isPro"):
        score += 25
    if _clean_text(candidate.get("ratingStatusSingles")) == "Rated":
        score += 15
    if _clean_text(candidate.get("thirdPartyRankings")):
        score += 5
    utr_value = _clean_float(candidate.get("singlesUtr")) or 0.0
    score += 1 if utr_value > 0 else 0
    return score, utr_value


def _search_rank_value(candidate: dict[str, Any]) -> int | None:
    direct_rank = _clean_int(candidate.get("utrRanking"))
    if direct_rank is not None:
        return direct_rank
    rankings = candidate.get("rankings")
    if not isinstance(rankings, list):
        return None
    ranks = [
        _clean_int(row.get("rank"))
        for row in rankings
        if isinstance(row, dict) and _clean_int(row.get("rank")) is not None
    ]
    return min(ranks) if ranks else None


def _record_from_candidate(
    *,
    player_name: str,
    rating_date: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    utr_value = _clean_float(candidate.get("utr"))
    if utr_value is None:
        utr_value = _clean_float(candidate.get("singlesUtr"))
    return {
        "player_name": player_name,
        "rating_date": rating_date,
        "utr_singles": utr_value,
        "utr_rank": _clean_int(candidate.get("utrRanking")) or _search_rank_value(candidate),
        "three_month_rating": _clean_float(candidate.get("threeMonthRating")),
        "nationality": _clean_text(candidate.get("nationality")),
        "provider_player_id": _clean_text(candidate.get("id")) or _clean_text(candidate.get("profileId")),
    }


def _choose_best_search_candidate(
    player_name: str,
    *,
    ioc: str | None,
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    exact_matches = _exact_name_matches(player_name, candidates)
    if not exact_matches:
        return None
    ranked = sorted(
        exact_matches,
        key=lambda candidate: _score_search_candidate(candidate, ioc=ioc),
        reverse=True,
    )
    return ranked[0] if ranked else None


def scrape_public_utr_snapshot(
    *,
    output_dir: str | Path,
    rating_date: str | None = None,
    tracked_players: Sequence[dict[str, Any]] | None = None,
    gender: str = "m",
    tags: str = "Pro",
    top_count: int = DEFAULT_TOP_COUNT,
    search_missing: bool = True,
    search_top: int = DEFAULT_SEARCH_TOP,
    workers: int = DEFAULT_WORKERS,
    session: requests.Session | None = None,
) -> tuple[Path, UtrPublicScrapeSummary]:
    snapshot_date = rating_date or date.today().isoformat()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    active_session = _build_session(session)
    summary = UtrPublicScrapeSummary()

    top_players = fetch_public_top_players(
        gender=gender,
        tags=tags,
        count=top_count,
        session=active_session,
    )
    summary.top_rankings_rows = len(top_players)

    tracked = [
        {
            "player_id": _clean_text(player.get("player_id")),
            "full_name": _clean_text(player.get("full_name")),
            "ioc": _clean_text(player.get("ioc")),
        }
        for player in (tracked_players or [])
        if _clean_text(player.get("full_name"))
    ]
    summary.tracked_players = len(tracked)

    if not tracked:
        rows = [
            _record_from_candidate(
                player_name=_clean_text(player.get("displayName")) or "",
                rating_date=snapshot_date,
                candidate=player,
            )
            for player in top_players
            if _clean_text(player.get("displayName"))
        ]
    else:
        top_index: dict[str, list[dict[str, Any]]] = {}
        for candidate in top_players:
            display_name = _clean_text(candidate.get("displayName"))
            if not display_name:
                continue
            top_index.setdefault(normalize_player_name(display_name), []).append(candidate)

        rows: list[dict[str, Any]] = []
        missing_players: list[dict[str, Any]] = []
        for player in tracked:
            lookup = normalize_player_name(player["full_name"] or "")
            candidates = top_index.get(lookup, [])
            matched = _choose_best_search_candidate(
                player["full_name"] or "",
                ioc=player.get("ioc"),
                candidates=candidates,
            )
            if matched is None:
                missing_players.append(player)
                continue
            rows.append(
                _record_from_candidate(
                    player_name=player["full_name"] or "",
                    rating_date=snapshot_date,
                    candidate=matched,
                )
            )
            summary.top_rankings_matches += 1

        if search_missing and missing_players:
            def scrape_missing_player(player: dict[str, Any]) -> dict[str, Any] | None:
                hits = fetch_public_player_search_hits(
                    player["full_name"] or "",
                    top=search_top,
                    session=_build_session(),
                )
                matched = _choose_best_search_candidate(
                    player["full_name"] or "",
                    ioc=player.get("ioc"),
                    candidates=hits,
                )
                if matched is None:
                    return None
                return _record_from_candidate(
                    player_name=player["full_name"] or "",
                    rating_date=snapshot_date,
                    candidate=matched,
                )

            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                for record in executor.map(scrape_missing_player, missing_players):
                    if record is None:
                        continue
                    rows.append(record)
                    summary.search_matches += 1

        summary.unmatched_players = summary.tracked_players - len(rows)

    rows = [
        row
        for row in rows
        if row.get("player_name") and row.get("utr_singles") is not None
    ]
    rows.sort(
        key=lambda row: (
            _clean_int(row.get("utr_rank")) is None,
            _clean_int(row.get("utr_rank")) or 10**9,
            normalize_player_name(_clean_text(row.get("player_name")) or ""),
        )
    )
    output_csv = output_path / f"utr_site_{tags.lower()}_{gender}_{snapshot_date}.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary.rows_written = len(rows)
    summary.output_csv = str(output_csv)
    return output_csv, summary
