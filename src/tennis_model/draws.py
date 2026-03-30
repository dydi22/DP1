from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


GENERIC_PLACEHOLDERS = {"qualifier", "tba", "alternate", "winner", "loser", "to be announced"}
MATCH_TYPE_TERMINAL_LABEL = {
    "qualifiersingles": "Qualified",
    "qualifying_singles": "Qualified",
}


@dataclass
class DrawEntry:
    player_name: str
    raw_name: str
    profile_url: str | None
    seed_tag: str | None
    is_bye: bool
    is_placeholder: bool


@dataclass
class ATPDraw:
    source_url: str
    match_type: str
    tournament_title: str
    overview_url: str | None
    tournament_slug: str | None
    tournament_code: str | None
    location_text: str | None
    date_text: str | None
    tournament_date: str | None
    draw_pdf_url: str | None
    draw_size: int
    round_labels: list[str]
    first_round_label: str
    terminal_label: str
    inferred_tourney_level: str | None
    inferred_best_of: int | None
    first_round_pairs: list[tuple[str, str]]
    placeholder_names: list[str]

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["first_round_pairs"] = [list(pair) for pair in self.first_round_pairs]
        return data


def _append_match_type(url: str, match_type: str | None) -> str:
    if not match_type:
        return url
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["matchtype"] = [match_type]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _name_from_slug(profile_url: str) -> str:
    parts = [part for part in profile_url.split("/") if part]
    if len(parts) < 3:
        return profile_url
    slug = parts[2]
    return " ".join(part.capitalize() for part in slug.split("-"))


def _infer_tourney_level(title: str) -> str | None:
    lowered = title.lower()
    if "masters 1000" in lowered:
        return "M"
    if "atp 500" in lowered or "atp 250" in lowered:
        return "A"
    if "finals" in lowered:
        return "F"
    if "olympic" in lowered:
        return "O"
    grand_slam_markers = {"australian open", "roland garros", "wimbledon", "us open"}
    if any(marker in lowered for marker in grand_slam_markers):
        return "G"
    return None


def _infer_tourney_level_from_badge_src(badge_src: str | None) -> str | None:
    lowered = _clean_text(badge_src).lower()
    if "1000" in lowered:
        return "M"
    if "500" in lowered or "250" in lowered:
        return "A"
    if "final" in lowered:
        return "F"
    return None


def _infer_best_of(match_type: str, tourney_level: str | None) -> int | None:
    if match_type in MATCH_TYPE_TERMINAL_LABEL:
        return 3
    if tourney_level == "G":
        return 5
    if tourney_level is not None:
        return 3
    return None


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


def _terminal_label(match_type: str, round_labels: list[str]) -> str:
    return MATCH_TYPE_TERMINAL_LABEL.get(match_type, "Champion")


def _parse_tournament_date(date_text: str | None) -> str | None:
    if not date_text:
        return None
    cleaned = _clean_text(date_text).replace(" - ", "-")
    patterns = [
        r"(?P<start>\d{1,2})-(?P<end>\d{1,2}) (?P<month>[A-Za-z]+), (?P<year>\d{4})",
        r"(?P<start>\d{1,2})-(?P<end>\d{1,2}) (?P<month>[A-Za-z]+) (?P<year>\d{4})",
    ]
    for pattern in patterns:
        match = pd.Series([cleaned]).str.extract(pattern).iloc[0]
        if match.notna().all():
            parsed = pd.to_datetime(
                f"{match['start']} {match['month']} {match['year']}",
                errors="coerce",
            )
            if pd.notna(parsed):
                return parsed.date().isoformat()
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.notna(parsed):
        return parsed.date().isoformat()
    return None


def _entry_from_stats_item(
    stats_item: Tag,
    *,
    source_url: str,
    placeholder_counter: Counter[str],
) -> DrawEntry:
    name_node = stats_item.select_one(".player-info .name")
    if name_node is None:
        raise ValueError("Could not find player name inside ATP draw item.")

    profile_link = name_node.select_one("a")
    profile_url = None
    if profile_link is not None and profile_link.get("href"):
        profile_url = urljoin(source_url, profile_link["href"])
        player_name = _name_from_slug(profile_link["href"])
    else:
        raw_base_name = _clean_text(name_node.get_text(" ", strip=True))
        player_name = raw_base_name

    raw_name = _clean_text(name_node.get_text(" ", strip=True))
    seed_tag = _clean_text(" ".join(span.get_text(" ", strip=True) for span in name_node.select("span"))) or None

    normalized_base = raw_name.split("(")[0].strip() if raw_name else player_name
    cleaned_name = _clean_text(player_name if profile_url else normalized_base) or _clean_text(player_name)
    lowered = cleaned_name.lower()
    is_bye = lowered == "bye"
    is_placeholder = False
    if not profile_url and not is_bye:
        if lowered in GENERIC_PLACEHOLDERS or lowered.startswith("qualifier") or lowered.startswith("tba"):
            placeholder_counter[cleaned_name] += 1
            suffix = placeholder_counter[cleaned_name]
            cleaned_name = cleaned_name if any(char.isdigit() for char in cleaned_name) else f"{cleaned_name} {suffix}"
            is_placeholder = True

    return DrawEntry(
        player_name="BYE" if is_bye else cleaned_name,
        raw_name=raw_name or cleaned_name,
        profile_url=profile_url,
        seed_tag=seed_tag,
        is_bye=is_bye,
        is_placeholder=is_placeholder,
    )


def fetch_atp_draw(draw_url: str, match_type: str | None = None, timeout: int = 30) -> ATPDraw:
    resolved_url = _append_match_type(draw_url, match_type)
    response = requests.get(resolved_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    draw_root = soup.select_one(".atp_draw")
    if draw_root is None:
        raise ValueError("Could not find ATP draw markup on the page.")

    title_link = soup.select_one(".schedule .title a")
    tournament_title = _clean_text(title_link.get_text(" ", strip=True)) if title_link else _clean_text(soup.title.string if soup.title else "")
    badge_image = soup.select_one(".info .badge img")
    badge_title = _clean_text(badge_image.get("alt")) if badge_image is not None else ""
    badge_src = badge_image.get("src") if badge_image is not None else None
    overview_href = title_link.get("href") if title_link else None
    overview_url = urljoin(resolved_url, overview_href) if overview_href else None
    overview_parts = [part for part in (overview_href or "").split("/") if part]
    tournament_slug = overview_parts[2] if len(overview_parts) >= 4 else None
    tournament_code = overview_parts[3] if len(overview_parts) >= 4 else None

    spans = soup.select(".schedule .date-location span")
    location_text = _clean_text(spans[0].get_text(" ", strip=True)) if len(spans) >= 1 else None
    date_text = _clean_text(spans[1].get_text(" ", strip=True)) if len(spans) >= 2 else None
    tournament_date = _parse_tournament_date(date_text)

    pdf_link = draw_root.select_one(".draws-download__link")
    draw_pdf_url = urljoin(resolved_url, pdf_link["href"]) if pdf_link and pdf_link.get("href") else None

    query = parse_qs(urlparse(resolved_url).query)
    resolved_match_type = query.get("matchtype", ["singles"])[0]

    round_sections = draw_root.select(".draw")
    round_labels = [
        _normalize_round_label(_clean_text(section.select_one(".draw-header").get_text(" ", strip=True)))
        for section in round_sections
        if section.select_one(".draw-header") is not None
    ]
    if not round_labels:
        raise ValueError("Could not find any ATP draw rounds on the page.")

    first_round_content = draw_root.select_one(".draw.draw-round-1 .draw-content")
    if first_round_content is None:
        raise ValueError("Could not find the first round on the ATP draw page.")

    placeholder_counter: Counter[str] = Counter()
    first_round_pairs: list[tuple[str, str]] = []
    placeholder_names: list[str] = []
    for item in first_round_content.select(":scope > .draw-item"):
        entries = [
            _entry_from_stats_item(stats_item, source_url=resolved_url, placeholder_counter=placeholder_counter)
            for stats_item in item.select(":scope .stats-item")
        ]
        if len(entries) != 2:
            raise ValueError("Each ATP first-round draw item must contain exactly two players.")
        first_round_pairs.append((entries[0].player_name, entries[1].player_name))
        for entry in entries:
            if entry.is_placeholder:
                placeholder_names.append(entry.player_name)

    draw_size = len(first_round_pairs) * 2
    inferred_tourney_level = (
        _infer_tourney_level(tournament_title)
        or _infer_tourney_level(badge_title)
        or _infer_tourney_level_from_badge_src(badge_src)
    )

    return ATPDraw(
        source_url=resolved_url,
        match_type=resolved_match_type,
        tournament_title=tournament_title,
        overview_url=overview_url,
        tournament_slug=tournament_slug,
        tournament_code=tournament_code,
        location_text=location_text,
        date_text=date_text,
        tournament_date=tournament_date,
        draw_pdf_url=draw_pdf_url,
        draw_size=draw_size,
        round_labels=round_labels,
        first_round_label=round_labels[0],
        terminal_label=_terminal_label(resolved_match_type, round_labels),
        inferred_tourney_level=inferred_tourney_level,
        inferred_best_of=_infer_best_of(resolved_match_type, inferred_tourney_level),
        first_round_pairs=first_round_pairs,
        placeholder_names=placeholder_names,
    )
