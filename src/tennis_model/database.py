from __future__ import annotations

import csv
import duckdb
import hashlib
import math
import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from tennis_model.names import normalize_player_name
from tennis_model.utr import load_name_aliases

CANONICAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CANONICAL_UTC_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


PLAYER_COLUMNS = [
    "player_id",
    "full_name",
    "name_first",
    "name_last",
    "full_name_normalized",
    "hand",
    "dob",
    "ioc",
    "height",
    "wikidata_id",
    "source_file",
    "updated_at",
]

PLAYER_RANKING_COLUMNS = [
    "ranking_date",
    "player_id",
    "rank",
    "points",
    "source_file",
    "updated_at",
]

TOURNAMENT_LOCATION_COLUMNS = [
    "tournament_lookup",
    "tournament_name",
    "city",
    "country",
    "timezone_name",
    "utc_offset_hours",
    "latitude",
    "longitude",
    "source_file",
    "updated_at",
]

TOURNAMENT_COLUMNS = [
    "tournament_id",
    "canonical_tournament_name",
    "tour_level",
    "default_surface",
    "default_indoor",
    "country",
    "city",
    "source_file",
    "updated_at",
]

TOURNAMENT_EDITION_COLUMNS = [
    "edition_id",
    "tournament_id",
    "season_year",
    "source_tourney_id",
    "tournament_name",
    "start_date",
    "end_date",
    "city",
    "country",
    "timezone_name",
    "latitude",
    "longitude",
    "altitude_m",
    "surface",
    "indoor",
    "draw_size",
    "source_file",
    "updated_at",
]

PLAYER_ALIAS_COLUMNS = [
    "alias_lookup",
    "alias_name",
    "canonical_player_id",
    "canonical_player_name",
    "source_system",
    "confidence_score",
    "source_file",
    "updated_at",
]

HISTORICAL_MATCH_COLUMNS = [
    "match_key",
    "source_file",
    "source_year",
    "tourney_id",
    "tourney_name",
    "surface",
    "indoor",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "score",
    "best_of",
    "round_name",
    "minutes",
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
    "ingested_at",
]

LIVE_MATCH_COLUMNS = [
    "feed_key",
    "match_id",
    "payload_hash",
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
    "p1_ace",
    "p1_df",
    "p1_svpt",
    "p1_1stIn",
    "p1_1stWon",
    "p1_2ndWon",
    "p1_bpSaved",
    "p1_bpFaced",
    "p2_ace",
    "p2_df",
    "p2_svpt",
    "p2_1stIn",
    "p2_1stWon",
    "p2_2ndWon",
    "p2_bpSaved",
    "p2_bpFaced",
    "raw_payload_json",
    "first_seen_at",
    "last_seen_at",
    "last_run_id",
]

LIVE_STAT_COLUMNS = [
    "p1_ace",
    "p1_df",
    "p1_svpt",
    "p1_1stIn",
    "p1_1stWon",
    "p1_2ndWon",
    "p1_bpSaved",
    "p1_bpFaced",
    "p2_ace",
    "p2_df",
    "p2_svpt",
    "p2_1stIn",
    "p2_1stWon",
    "p2_2ndWon",
    "p2_bpSaved",
    "p2_bpFaced",
]

PLAYER_UTR_COLUMNS = [
    "player_lookup",
    "player_name",
    "canonical_player_name",
    "canonical_player_id",
    "rating_date",
    "utr_singles",
    "utr_rank",
    "three_month_rating",
    "nationality",
    "provider_player_id",
    "source_file",
    "source_type",
    "updated_at",
]

FLASHSCORE_MATCH_COLUMNS = [
    "match_key",
    "event_id",
    "payload_hash",
    "match_url",
    "home_name",
    "away_name",
    "home_participant_id",
    "away_participant_id",
    "tournament_name",
    "category_name",
    "country_name",
    "point_score_home",
    "point_score_away",
    "current_game_feed_raw",
    "current_game_feed_json",
    "common_feed_raw",
    "common_feed_json",
    "match_history_feed_raw",
    "game_feed_raw",
    "raw_payload_json",
    "first_seen_at",
    "last_seen_at",
    "last_run_id",
]

GRAND_SLAM_MATCH_COLUMNS = [
    "match_key",
    "slam_code",
    "provider_name",
    "source_match_id",
    "season_year",
    "payload_hash",
    "match_url",
    "event_name",
    "round_name",
    "court_name",
    "status_text",
    "status_code",
    "home_name",
    "away_name",
    "winner_side",
    "score_text",
    "point_score_home",
    "point_score_away",
    "discovery_source",
    "schedule_payload_json",
    "detail_payload_json",
    "stats_payload_json",
    "history_payload_json",
    "keys_payload_json",
    "insights_payload_json",
    "page_payload_json",
    "raw_payload_json",
    "first_seen_at",
    "last_seen_at",
    "last_run_id",
]

MATCH_FACT_COLUMNS = [
    "match_id",
    "source_match_key",
    "source_system",
    "tournament_id",
    "edition_id",
    "start_time_utc",
    "end_time_utc",
    "duration_minutes",
    "surface",
    "best_of",
    "tournament",
    "round",
    "score_text",
    "winner_player_id",
    "loser_player_id",
    "winner_name",
    "loser_name",
    "tourney_level",
    "updated_at",
]

PLAYER_MATCH_STATS_COLUMNS = [
    "match_id",
    "player_id",
    "opponent_id",
    "is_winner",
    "side",
    "ace",
    "df",
    "svpt",
    "first_in",
    "first_won",
    "second_won",
    "service_games",
    "bp_saved",
    "bp_faced",
    "minutes",
    "updated_at",
]

PLAYER_MATCH_LOAD_COLUMNS = [
    "match_id",
    "player_id",
    "opponent_id",
    "match_start_time_utc",
    "player_minutes_last_1_match",
    "player_minutes_last_3_matches",
    "player_minutes_last_7d",
    "player_minutes_last_14d",
    "player_avg_match_minutes_last_30d",
    "player_long_match_flag_last_match",
    "back_to_back_long_match_count",
    "player_matches_last_7d",
    "opponent_minutes_last_7d",
    "opponent_matches_last_7d",
    "minutes_diff_last_7d",
    "prev_tournament_city",
    "current_tournament_city",
    "prev_match_end_utc",
    "current_match_start_utc",
    "previous_utc_offset_hours",
    "current_utc_offset_hours",
    "timezones_crossed_signed",
    "travel_direction",
    "great_circle_km",
    "hours_between_matches",
    "days_rest",
    "back_to_back_week_flag",
    "tz_shift_signed",
    "tz_shift_abs",
    "eastward_shift_flag",
    "short_recovery_after_travel",
    "previous_match_duration",
    "travel_fatigue_score",
    "travel_timezones_last_7d",
    "tiebreaks_last_7d",
    "load_score",
    "load_plus_travel_score",
    "eastward_shift_minutes_last_7d",
    "eastward_shift_days_rest",
    "tz_shift_abs_previous_match_duration",
    "back_to_back_week_tz_shift_abs",
    "load_weight_minutes_last_7d",
    "load_weight_matches_last_7d",
    "load_weight_travel_timezones_last_7d",
    "load_weight_tiebreaks_last_7d",
    "updated_at",
]

POINT_EVENT_COLUMNS = [
    "match_id",
    "set_no",
    "game_no",
    "point_no",
    "server_id",
    "returner_id",
    "score_state",
    "break_point_flag",
    "ace_flag",
    "winner_flag",
    "unforced_error_flag",
    "rally_count",
    "serve_speed",
    "serve_direction",
    "return_depth",
    "point_winner_id",
    "source_system",
    "updated_at",
]

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS source_files (
    source_path TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    modified_time_ns BIGINT NOT NULL,
    synced_at TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pipeline_metadata (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    full_name TEXT,
    name_first TEXT,
    name_last TEXT,
    full_name_normalized TEXT,
    hand TEXT,
    dob TEXT,
    ioc TEXT,
    height INTEGER,
    wikidata_id TEXT,
    latest_utr_singles REAL,
    latest_utr_rating_date TEXT,
    latest_utr_rank INTEGER,
    latest_utr_three_month_rating REAL,
    latest_utr_nationality TEXT,
    latest_utr_provider_player_id TEXT,
    source_file TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_players_full_name_normalized
    ON players (full_name_normalized);

CREATE TABLE IF NOT EXISTS player_aliases (
    alias_lookup TEXT NOT NULL,
    alias_name TEXT NOT NULL,
    canonical_player_id TEXT NOT NULL,
    canonical_player_name TEXT NOT NULL,
    source_system TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    source_file TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (alias_lookup, canonical_player_id, source_system)
);

CREATE INDEX IF NOT EXISTS idx_player_aliases_canonical
    ON player_aliases (canonical_player_id, source_system);

CREATE TABLE IF NOT EXISTS player_utr_ratings (
    player_lookup TEXT NOT NULL,
    player_name TEXT NOT NULL,
    canonical_player_name TEXT NOT NULL,
    canonical_player_id TEXT,
    rating_date TEXT NOT NULL,
    utr_singles REAL NOT NULL,
    utr_rank INTEGER,
    three_month_rating REAL,
    nationality TEXT,
    provider_player_id TEXT,
    source_file TEXT NOT NULL,
    source_type TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (player_lookup, rating_date)
);

CREATE INDEX IF NOT EXISTS idx_player_utr_ratings_rating_date
    ON player_utr_ratings (rating_date);

CREATE TABLE IF NOT EXISTS player_rankings (
    ranking_date TEXT NOT NULL,
    player_id TEXT NOT NULL,
    rank INTEGER,
    points INTEGER,
    source_file TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (ranking_date, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_rankings_player_date
    ON player_rankings (player_id, ranking_date);

CREATE TABLE IF NOT EXISTS tournament_locations (
    tournament_lookup TEXT PRIMARY KEY,
    tournament_name TEXT NOT NULL,
    city TEXT,
    country TEXT,
    timezone_name TEXT,
    utc_offset_hours REAL,
    latitude REAL,
    longitude REAL,
    source_file TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tournament_locations_name
    ON tournament_locations (tournament_name);

CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id TEXT PRIMARY KEY,
    canonical_tournament_name TEXT NOT NULL,
    tour_level TEXT,
    default_surface TEXT,
    default_indoor TEXT,
    country TEXT,
    city TEXT,
    source_file TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tournaments_name
    ON tournaments (canonical_tournament_name);

CREATE TABLE IF NOT EXISTS tournament_editions (
    edition_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    season_year INTEGER,
    source_tourney_id TEXT,
    tournament_name TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT,
    city TEXT,
    country TEXT,
    timezone_name TEXT,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    surface TEXT,
    indoor TEXT,
    draw_size INTEGER,
    source_file TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tournament_editions_tournament
    ON tournament_editions (tournament_id, season_year);

CREATE TABLE IF NOT EXISTS historical_matches (
    match_key TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    source_year INTEGER,
    tourney_id TEXT,
    tourney_name TEXT,
    surface TEXT,
    indoor TEXT,
    draw_size INTEGER,
    tourney_level TEXT,
    tourney_date TEXT,
    match_num INTEGER,
    winner_id TEXT,
    winner_seed TEXT,
    winner_entry TEXT,
    winner_name TEXT,
    winner_hand TEXT,
    winner_ht INTEGER,
    winner_ioc TEXT,
    winner_age REAL,
    winner_rank INTEGER,
    winner_rank_points INTEGER,
    loser_id TEXT,
    loser_seed TEXT,
    loser_entry TEXT,
    loser_name TEXT,
    loser_hand TEXT,
    loser_ht INTEGER,
    loser_ioc TEXT,
    loser_age REAL,
    loser_rank INTEGER,
    loser_rank_points INTEGER,
    score TEXT,
    best_of INTEGER,
    round_name TEXT,
    minutes INTEGER,
    w_ace INTEGER,
    w_df INTEGER,
    w_svpt INTEGER,
    w_1stIn INTEGER,
    w_1stWon INTEGER,
    w_2ndWon INTEGER,
    w_SvGms INTEGER,
    w_bpSaved INTEGER,
    w_bpFaced INTEGER,
    l_ace INTEGER,
    l_df INTEGER,
    l_svpt INTEGER,
    l_1stIn INTEGER,
    l_1stWon INTEGER,
    l_2ndWon INTEGER,
    l_SvGms INTEGER,
    l_bpSaved INTEGER,
    l_bpFaced INTEGER,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_historical_matches_tourney_date
    ON historical_matches (tourney_date);

CREATE INDEX IF NOT EXISTS idx_historical_matches_players
    ON historical_matches (winner_id, loser_id, tourney_date);

CREATE TABLE IF NOT EXISTS live_match_feed (
    feed_key TEXT PRIMARY KEY,
    match_id TEXT,
    payload_hash TEXT NOT NULL,
    match_date TEXT,
    round_text TEXT,
    round_name TEXT,
    court_label TEXT,
    duration_text TEXT,
    player_1 TEXT,
    player_2 TEXT,
    winner TEXT,
    score_text TEXT,
    results_url TEXT,
    stats_url TEXT,
    h2h_url TEXT,
    surface TEXT,
    best_of INTEGER,
    tourney_level TEXT,
    stats_fetch_error TEXT,
    p1_ace INTEGER,
    p1_df INTEGER,
    p1_svpt INTEGER,
    p1_1stIn INTEGER,
    p1_1stWon INTEGER,
    p1_2ndWon INTEGER,
    p1_bpSaved INTEGER,
    p1_bpFaced INTEGER,
    p2_ace INTEGER,
    p2_df INTEGER,
    p2_svpt INTEGER,
    p2_1stIn INTEGER,
    p2_1stWon INTEGER,
    p2_2ndWon INTEGER,
    p2_bpSaved INTEGER,
    p2_bpFaced INTEGER,
    raw_payload_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_run_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_live_match_feed_match_date
    ON live_match_feed (match_date);

CREATE INDEX IF NOT EXISTS idx_live_match_feed_players
    ON live_match_feed (player_1, player_2, match_date);

CREATE TABLE IF NOT EXISTS live_match_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    feed_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    UNIQUE (feed_key, payload_hash)
);

CREATE TABLE IF NOT EXISTS flashscore_match_feed (
    match_key TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    match_url TEXT NOT NULL,
    home_name TEXT,
    away_name TEXT,
    home_participant_id TEXT,
    away_participant_id TEXT,
    tournament_name TEXT,
    category_name TEXT,
    country_name TEXT,
    point_score_home TEXT,
    point_score_away TEXT,
    current_game_feed_raw TEXT NOT NULL,
    current_game_feed_json TEXT NOT NULL,
    common_feed_raw TEXT NOT NULL,
    common_feed_json TEXT NOT NULL,
    match_history_feed_raw TEXT,
    game_feed_raw TEXT,
    raw_payload_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_run_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_flashscore_match_feed_event_id
    ON flashscore_match_feed (event_id);

CREATE INDEX IF NOT EXISTS idx_flashscore_match_feed_players
    ON flashscore_match_feed (home_name, away_name);

CREATE TABLE IF NOT EXISTS flashscore_point_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    match_key TEXT NOT NULL,
    event_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    point_score_home TEXT,
    point_score_away TEXT,
    current_game_feed_raw TEXT NOT NULL,
    current_game_feed_json TEXT NOT NULL,
    common_feed_raw TEXT NOT NULL,
    common_feed_json TEXT NOT NULL,
    match_history_feed_raw TEXT,
    game_feed_raw TEXT,
    captured_at TEXT NOT NULL,
    UNIQUE (match_key, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_flashscore_point_snapshots_match_key
    ON flashscore_point_snapshots (match_key, captured_at);

CREATE TABLE IF NOT EXISTS grand_slam_match_feed (
    match_key TEXT PRIMARY KEY,
    slam_code TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    season_year INTEGER,
    payload_hash TEXT NOT NULL,
    match_url TEXT,
    event_name TEXT,
    round_name TEXT,
    court_name TEXT,
    status_text TEXT,
    status_code TEXT,
    home_name TEXT,
    away_name TEXT,
    winner_side TEXT,
    score_text TEXT,
    point_score_home TEXT,
    point_score_away TEXT,
    discovery_source TEXT,
    schedule_payload_json TEXT NOT NULL,
    detail_payload_json TEXT NOT NULL,
    stats_payload_json TEXT NOT NULL,
    history_payload_json TEXT NOT NULL,
    keys_payload_json TEXT NOT NULL,
    insights_payload_json TEXT NOT NULL,
    page_payload_json TEXT NOT NULL,
    raw_payload_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_run_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_grand_slam_match_feed_slam_source
    ON grand_slam_match_feed (slam_code, season_year, source_match_id);

CREATE INDEX IF NOT EXISTS idx_grand_slam_match_feed_players
    ON grand_slam_match_feed (home_name, away_name);

CREATE TABLE IF NOT EXISTS grand_slam_match_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    match_key TEXT NOT NULL,
    slam_code TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    status_text TEXT,
    score_text TEXT,
    point_score_home TEXT,
    point_score_away TEXT,
    schedule_payload_json TEXT NOT NULL,
    detail_payload_json TEXT NOT NULL,
    stats_payload_json TEXT NOT NULL,
    history_payload_json TEXT NOT NULL,
    keys_payload_json TEXT NOT NULL,
    insights_payload_json TEXT NOT NULL,
    page_payload_json TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    UNIQUE (match_key, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_grand_slam_match_snapshots_match_key
    ON grand_slam_match_snapshots (match_key, captured_at);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    source_match_key TEXT NOT NULL UNIQUE,
    source_system TEXT NOT NULL,
    start_time_utc TEXT NOT NULL,
    end_time_utc TEXT,
    duration_minutes INTEGER,
    surface TEXT,
    best_of INTEGER,
    tournament TEXT,
    round TEXT,
    score_text TEXT,
    winner_player_id TEXT,
    loser_player_id TEXT,
    winner_name TEXT,
    loser_name TEXT,
    tourney_level TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_matches_start_time
    ON matches (start_time_utc);

CREATE INDEX IF NOT EXISTS idx_matches_players
    ON matches (winner_player_id, loser_player_id);

CREATE TABLE IF NOT EXISTS player_match_stats (
    match_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    opponent_id TEXT,
    is_winner INTEGER NOT NULL,
    side TEXT NOT NULL,
    ace INTEGER,
    df INTEGER,
    svpt INTEGER,
    first_in INTEGER,
    first_won INTEGER,
    second_won INTEGER,
    service_games INTEGER,
    bp_saved INTEGER,
    bp_faced INTEGER,
    minutes INTEGER,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (match_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_match_stats_player_time
    ON player_match_stats (player_id, match_id);

CREATE TABLE IF NOT EXISTS player_match_load_features (
    match_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    opponent_id TEXT,
    match_start_time_utc TEXT NOT NULL,
    player_minutes_last_1_match INTEGER,
    player_minutes_last_3_matches INTEGER NOT NULL DEFAULT 0,
    player_minutes_last_7d INTEGER NOT NULL DEFAULT 0,
    player_minutes_last_14d INTEGER NOT NULL DEFAULT 0,
    player_avg_match_minutes_last_30d REAL,
    player_long_match_flag_last_match INTEGER NOT NULL DEFAULT 0,
    back_to_back_long_match_count INTEGER NOT NULL DEFAULT 0,
    player_matches_last_7d INTEGER NOT NULL DEFAULT 0,
    opponent_minutes_last_7d INTEGER NOT NULL DEFAULT 0,
    opponent_matches_last_7d INTEGER NOT NULL DEFAULT 0,
    minutes_diff_last_7d INTEGER NOT NULL DEFAULT 0,
    prev_tournament_city TEXT,
    current_tournament_city TEXT,
    prev_match_end_utc TEXT,
    current_match_start_utc TEXT,
    previous_utc_offset_hours REAL,
    current_utc_offset_hours REAL,
    timezones_crossed_signed REAL,
    travel_direction TEXT,
    great_circle_km REAL,
    hours_between_matches REAL,
    days_rest REAL,
    back_to_back_week_flag INTEGER NOT NULL DEFAULT 0,
    tz_shift_signed REAL,
    tz_shift_abs REAL,
    eastward_shift_flag INTEGER NOT NULL DEFAULT 0,
    short_recovery_after_travel INTEGER NOT NULL DEFAULT 0,
    previous_match_duration INTEGER,
    travel_fatigue_score REAL NOT NULL DEFAULT 0,
    travel_timezones_last_7d REAL NOT NULL DEFAULT 0,
    tiebreaks_last_7d INTEGER NOT NULL DEFAULT 0,
    load_score REAL NOT NULL DEFAULT 0,
    load_plus_travel_score REAL NOT NULL DEFAULT 0,
    eastward_shift_minutes_last_7d REAL NOT NULL DEFAULT 0,
    eastward_shift_days_rest REAL NOT NULL DEFAULT 0,
    tz_shift_abs_previous_match_duration REAL NOT NULL DEFAULT 0,
    back_to_back_week_tz_shift_abs REAL NOT NULL DEFAULT 0,
    load_weight_minutes_last_7d REAL NOT NULL,
    load_weight_matches_last_7d REAL NOT NULL,
    load_weight_travel_timezones_last_7d REAL NOT NULL,
    load_weight_tiebreaks_last_7d REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (match_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_match_load_features_player_time
    ON player_match_load_features (player_id, match_start_time_utc);

CREATE TABLE IF NOT EXISTS point_events (
    point_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    set_no INTEGER NOT NULL,
    game_no INTEGER NOT NULL,
    point_no INTEGER NOT NULL,
    server_id TEXT,
    returner_id TEXT,
    score_state TEXT,
    break_point_flag INTEGER NOT NULL DEFAULT 0,
    ace_flag INTEGER NOT NULL DEFAULT 0,
    winner_flag INTEGER NOT NULL DEFAULT 0,
    unforced_error_flag INTEGER NOT NULL DEFAULT 0,
    rally_count INTEGER,
    serve_speed INTEGER,
    serve_direction TEXT,
    return_depth TEXT,
    point_winner_id TEXT,
    source_system TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (match_id, set_no, game_no, point_no, source_system)
);

CREATE INDEX IF NOT EXISTS idx_point_events_match
    ON point_events (match_id, set_no, game_no, point_no);

CREATE INDEX IF NOT EXISTS idx_point_events_server_returner
    ON point_events (server_id, returner_id);

"""

UNIFIED_MATCH_STREAM_VIEW_SQL = """
CREATE VIEW unified_match_stream AS
SELECT
    'historical_main_tour' AS match_source,
    match_key AS record_key,
    tourney_date AS match_date,
    tourney_name,
    surface,
    round_name,
    winner_name AS player_1_name,
    loser_name AS player_2_name,
    winner_name,
    score AS score_text,
    minutes,
    best_of,
    tourney_level,
    NULL AS results_url,
    NULL AS stats_url,
    ingested_at AS updated_at
FROM historical_matches
UNION ALL
SELECT
    'live_atp_results' AS match_source,
    feed_key AS record_key,
    match_date,
    NULL AS tourney_name,
    surface,
    round_name,
    player_1 AS player_1_name,
    player_2 AS player_2_name,
    winner AS winner_name,
    score_text,
    NULL AS minutes,
    best_of,
    tourney_level,
    results_url,
    stats_url,
    last_seen_at AS updated_at
FROM live_match_feed
UNION ALL
SELECT
    'live_flashscore_points' AS match_source,
    match_key AS record_key,
    NULL AS match_date,
    tournament_name,
    NULL AS surface,
    NULL AS round_name,
    home_name AS player_1_name,
    away_name AS player_2_name,
    NULL AS winner_name,
    CASE
        WHEN point_score_home IS NULL AND point_score_away IS NULL THEN NULL
        ELSE COALESCE(point_score_home, '') || '-' || COALESCE(point_score_away, '')
    END AS score_text,
    NULL AS minutes,
    NULL AS best_of,
    NULL AS tourney_level,
    match_url AS results_url,
    NULL AS stats_url,
    last_seen_at AS updated_at
FROM flashscore_match_feed
UNION ALL
SELECT
    'grand_slam_official' AS match_source,
    match_key AS record_key,
    NULL AS match_date,
    slam_code || COALESCE(' ' || CAST(season_year AS TEXT), '') AS tourney_name,
    NULL AS surface,
    round_name,
    home_name AS player_1_name,
    away_name AS player_2_name,
    CASE
        WHEN winner_side = '1' THEN home_name
        WHEN winner_side = '2' THEN away_name
        ELSE NULL
    END AS winner_name,
    score_text,
    NULL AS minutes,
    NULL AS best_of,
    slam_code AS tourney_level,
    match_url AS results_url,
    NULL AS stats_url,
    last_seen_at AS updated_at
FROM grand_slam_match_feed
"""


@dataclass
class SyncSummary:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_files: int = 0
    synced_files: int = 0
    rows_read: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped_files": self.skipped_files,
            "synced_files": self.synced_files,
            "rows_read": self.rows_read,
        }


class CursorWrapper:
    def __init__(self, cursor: Any, *, backend: str) -> None:
        self._cursor = cursor
        self._backend = backend

    def _description_columns(self) -> list[str]:
        description = getattr(self._cursor, "description", None) or []
        return [str(column[0]) for column in description]

    def _wrap_row(self, row: Any) -> Any:
        if row is None or self._backend == "sqlite":
            return row
        columns = self._description_columns()
        return {column: row[index] for index, column in enumerate(columns)}

    def fetchone(self) -> Any:
        return self._wrap_row(self._cursor.fetchone())

    def fetchall(self) -> list[Any]:
        return [self._wrap_row(row) for row in self._cursor.fetchall()]

    def __iter__(self) -> Iterable[Any]:
        if self._backend == "sqlite":
            return iter(self._cursor)
        return iter(self.fetchall())


class ConnectionWrapper:
    def __init__(self, connection: Any, *, backend: str) -> None:
        self._connection = connection
        self.backend = backend

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> CursorWrapper:
        cursor = self._connection.execute(sql, parameters or [])
        return CursorWrapper(cursor, backend=self.backend)

    def executemany(self, sql: str, seq_of_parameters: Iterable[Sequence[Any]]) -> Any:
        return self._connection.executemany(sql, list(seq_of_parameters))

    def executescript(self, script: str) -> None:
        if self.backend == "sqlite":
            self._connection.executescript(script)
            return
        statements = [statement.strip() for statement in script.split(";") if statement.strip()]
        for statement in statements:
            self._connection.execute(statement)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ConnectionWrapper":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        except Exception:
            pass
        self.close()


def _utc_now() -> str:
    return _format_utc_datetime(datetime.now(timezone.utc))


def _duckdb_schema_sql() -> str:
    schema_sql = "\n".join(
        line for line in SCHEMA_SQL.splitlines() if not line.strip().upper().startswith("PRAGMA ")
    )
    replacements = {
        "snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT": (
            "snapshot_id BIGINT PRIMARY KEY DEFAULT nextval('snapshot_id_seq')"
        ),
        "point_event_id INTEGER PRIMARY KEY AUTOINCREMENT": (
            "point_event_id BIGINT PRIMARY KEY DEFAULT nextval('point_event_id_seq')"
        ),
    }
    for source, target in replacements.items():
        schema_sql = schema_sql.replace(source, target)
    sequence_sql = """
CREATE SEQUENCE IF NOT EXISTS snapshot_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS point_event_id_seq START 1;
"""
    return sequence_sql + "\n" + schema_sql


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _clean_identifier(value: Any) -> str | None:
    cleaned = _clean_text(value)
    return cleaned if cleaned not in {"nan", "NaN"} else None


def _clean_int(value: Any) -> int | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _clean_float(value: Any) -> float | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_date_value(value: date) -> str:
    return value.isoformat()


def _format_utc_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_date(value: Any, *, input_format: str | None = None) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    if CANONICAL_DATE_RE.fullmatch(cleaned):
        return cleaned
    if input_format:
        try:
            return _format_date_value(datetime.strptime(cleaned, input_format).date())
        except ValueError:
            return None
    try:
        parsed = pd.to_datetime(cleaned, errors="coerce")
        if pd.isna(parsed):
            return None
        return _format_date_value(parsed.date())
    except Exception:
        return None


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _payload_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _parse_utc_datetime(value: str | None) -> datetime | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    normalized = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_utc_datetime(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    if CANONICAL_UTC_DATETIME_RE.fullmatch(cleaned):
        return cleaned
    parsed = _parse_utc_datetime(cleaned)
    if parsed is not None:
        return _format_utc_datetime(parsed)
    try:
        parsed_any = pd.to_datetime(cleaned, errors="coerce", utc=True)
        if pd.isna(parsed_any):
            return None
        python_dt = parsed_any.to_pydatetime()
        return _format_utc_datetime(python_dt)
    except Exception:
        return None


def _upsert_sql(table: str, columns: Sequence[str], conflict_columns: Sequence[str]) -> str:
    insert_columns = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in columns
        if column not in set(conflict_columns)
    )
    conflict = ", ".join(conflict_columns)
    return (
        f"INSERT INTO {table} ({insert_columns}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )


def _batched(iterable: Iterable[tuple[Any, ...]], batch_size: int = 1000) -> Iterable[list[tuple[Any, ...]]]:
    batch: list[tuple[Any, ...]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _historical_match_key(row: dict[str, str]) -> str:
    tourney_id = _clean_identifier(row.get("tourney_id")) or "unknown_tournament"
    tourney_date = _clean_date(row.get("tourney_date"), input_format="%Y%m%d") or "unknown_date"
    match_num = _clean_int(row.get("match_num")) or 0
    round_name = _clean_text(row.get("round")) or "unknown_round"
    winner_id = _clean_identifier(row.get("winner_id"))
    loser_id = _clean_identifier(row.get("loser_id"))
    if winner_id and loser_id:
        player_key = f"{winner_id}|{loser_id}"
    else:
        player_key = "|".join(
            [
                normalize_player_name(row.get("winner_name", "")),
                normalize_player_name(row.get("loser_name", "")),
            ]
        )
    return f"{tourney_id}|{tourney_date}|{match_num}|{round_name}|{player_key}"


def _live_feed_key(row: dict[str, Any]) -> str:
    match_id = _clean_identifier(row.get("match_id"))
    if match_id:
        return match_id.upper()
    match_date = _clean_date(row.get("match_date")) or "unknown_date"
    round_name = _clean_text(row.get("round_name")) or "unknown_round"
    players = sorted(
        [
            normalize_player_name(row.get("player_1", "")),
            normalize_player_name(row.get("player_2", "")),
        ]
    )
    return f"{match_date}|{round_name}|{players[0]}|{players[1]}"


def _flashscore_match_key(event_id: Any, match_url: Any) -> str:
    cleaned_event_id = _clean_identifier(event_id)
    if cleaned_event_id:
        return cleaned_event_id
    cleaned_match_url = _clean_text(match_url)
    if not cleaned_match_url:
        raise ValueError("Flashscore snapshot is missing both event_id and match_url")
    return cleaned_match_url


def _grand_slam_match_key(slam_code: Any, source_match_id: Any, match_url: Any) -> str:
    cleaned_slam_code = _clean_text(slam_code) or "grand_slam"
    cleaned_source_match_id = _clean_identifier(source_match_id)
    if cleaned_source_match_id:
        return f"{cleaned_slam_code}:{cleaned_source_match_id}"
    cleaned_match_url = _clean_text(match_url)
    if cleaned_match_url:
        return f"{cleaned_slam_code}:{cleaned_match_url}"
    raise ValueError("Grand Slam snapshot is missing both source_match_id and match_url")


def _source_year_from_path(path: Path) -> int | None:
    match = re.search(r"atp_matches_(\d{4})", path.name)
    return int(match.group(1)) if match else None


def _resolve_main_tour_files(data_dir: Path, years: Sequence[str] | None, include_ongoing: bool) -> list[Path]:
    files: list[Path] = []
    if years:
        files.extend(data_dir / f"atp_matches_{year}.csv" for year in years)
    else:
        files.extend(sorted(data_dir.glob("atp_matches_[0-9][0-9][0-9][0-9].csv")))
    if include_ongoing:
        files.extend(sorted(data_dir.glob("atp_matches_[0-9][0-9][0-9][0-9]_ongoing.csv")))
    resolved = [path for path in files if path.exists()]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ATP match files: {', '.join(missing)}")
    return sorted(dict.fromkeys(resolved))


def _historical_match_start_time(value: str | None) -> str | None:
    cleaned = _clean_date(value)
    if cleaned is None:
        return None
    parsed = datetime.combine(datetime.fromisoformat(cleaned).date(), time.min, tzinfo=timezone.utc)
    return _format_utc_datetime(parsed)


def _match_end_time(start_time_utc: str | None, duration_minutes: Any) -> str | None:
    start_dt = _parse_utc_datetime(start_time_utc)
    minutes = _clean_int(duration_minutes)
    if start_dt is None or minutes is None:
        return None
    return _format_utc_datetime(start_dt + timedelta(minutes=minutes))


def _count_tiebreaks(score_text: Any) -> int:
    cleaned = _clean_text(score_text)
    if cleaned is None:
        return 0
    return len(re.findall(r"(?:7-6|6-7)", cleaned))


def _player_reference(player_id: Any, player_name: Any) -> str | None:
    cleaned_id = _clean_identifier(player_id)
    if cleaned_id:
        return cleaned_id
    cleaned_name = _clean_text(player_name)
    if cleaned_name:
        return f"name::{normalize_player_name(cleaned_name)}"
    return None


def _normalize_tournament_lookup(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_") or None


def _tournament_id(tourney_id: Any, tourney_name: Any, tourney_level: Any) -> str | None:
    cleaned_tourney_id = _clean_identifier(tourney_id)
    cleaned_level = _clean_text(tourney_level) or "unknown"
    if cleaned_tourney_id:
        match = re.match(r"^\d{4}-(.+)$", cleaned_tourney_id)
        if match:
            return f"{cleaned_level}:{match.group(1)}"
        return f"{cleaned_level}:{cleaned_tourney_id}"
    lookup = _normalize_tournament_lookup(tourney_name)
    if lookup is None:
        return None
    return f"{cleaned_level}:{lookup}"


def _tournament_edition_id(
    tourney_id: Any,
    tourney_name: Any,
    tourney_level: Any,
    tourney_date: Any,
) -> str | None:
    cleaned_tourney_id = _clean_identifier(tourney_id)
    if cleaned_tourney_id:
        return cleaned_tourney_id
    tournament_id = _tournament_id(tourney_id, tourney_name, tourney_level)
    cleaned_date = _clean_date(tourney_date)
    if tournament_id is None or cleaned_date is None:
        return None
    return f"{tournament_id}:{cleaned_date}"


def _default_tournament_city(tournament_name: Any) -> str | None:
    cleaned = _clean_text(tournament_name)
    return cleaned


def _haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _resolve_ranking_files(data_dir: Path, ranking_files: Sequence[str] | None) -> list[Path]:
    if ranking_files:
        files = [data_dir / name for name in ranking_files]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing ATP ranking files: {', '.join(missing)}")
        return files
    return sorted(data_dir.glob("atp_rankings*.csv"))


class TennisDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = "duckdb" if self.db_path.suffix.lower() == ".duckdb" else "sqlite"

    def connect(self) -> ConnectionWrapper:
        if self.backend == "duckdb":
            connection = duckdb.connect(str(self.db_path))
        else:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row
        return ConnectionWrapper(connection, backend=self.backend)

    def initialize(self) -> None:
        with self.connect() as connection:
            schema_sql = _duckdb_schema_sql() if self.backend == "duckdb" else SCHEMA_SQL
            connection.executescript(schema_sql)
            self._ensure_column(connection, "players", "latest_utr_singles", "REAL")
            self._ensure_column(connection, "players", "latest_utr_rating_date", "TEXT")
            self._ensure_column(connection, "players", "latest_utr_rank", "INTEGER")
            self._ensure_column(connection, "players", "latest_utr_three_month_rating", "REAL")
            self._ensure_column(connection, "players", "latest_utr_nationality", "TEXT")
            self._ensure_column(connection, "players", "latest_utr_provider_player_id", "TEXT")
            self._ensure_column(connection, "player_utr_ratings", "utr_rank", "INTEGER")
            self._ensure_column(connection, "player_utr_ratings", "three_month_rating", "REAL")
            self._ensure_column(connection, "player_utr_ratings", "nationality", "TEXT")
            self._ensure_column(connection, "player_utr_ratings", "provider_player_id", "TEXT")
            self._ensure_column(connection, "player_utr_ratings", "canonical_player_id", "TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_utr_ratings_canonical_player_date
                    ON player_utr_ratings (canonical_player_id, rating_date)
                """
            )
            self._ensure_column(connection, "matches", "tournament_id", "TEXT")
            self._ensure_column(connection, "matches", "edition_id", "TEXT")
            self._ensure_column(connection, "player_match_load_features", "prev_tournament_city", "TEXT")
            self._ensure_column(connection, "player_match_load_features", "current_tournament_city", "TEXT")
            self._ensure_column(connection, "player_match_load_features", "prev_match_end_utc", "TEXT")
            self._ensure_column(connection, "player_match_load_features", "current_match_start_utc", "TEXT")
            self._ensure_column(connection, "player_match_load_features", "previous_utc_offset_hours", "REAL")
            self._ensure_column(connection, "player_match_load_features", "current_utc_offset_hours", "REAL")
            self._ensure_column(connection, "player_match_load_features", "timezones_crossed_signed", "REAL")
            self._ensure_column(connection, "player_match_load_features", "travel_direction", "TEXT")
            self._ensure_column(connection, "player_match_load_features", "great_circle_km", "REAL")
            self._ensure_column(connection, "player_match_load_features", "hours_between_matches", "REAL")
            self._ensure_column(connection, "player_match_load_features", "days_rest", "REAL")
            self._ensure_column(connection, "player_match_load_features", "back_to_back_week_flag", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "tz_shift_signed", "REAL")
            self._ensure_column(connection, "player_match_load_features", "tz_shift_abs", "REAL")
            self._ensure_column(connection, "player_match_load_features", "eastward_shift_flag", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "short_recovery_after_travel", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "previous_match_duration", "INTEGER")
            self._ensure_column(connection, "player_match_load_features", "travel_fatigue_score", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "load_plus_travel_score", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "eastward_shift_minutes_last_7d", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "eastward_shift_days_rest", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "tz_shift_abs_previous_match_duration", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(connection, "player_match_load_features", "back_to_back_week_tz_shift_abs", "REAL NOT NULL DEFAULT 0")
            self._maybe_normalize_temporal_columns(connection)
            self._refresh_utr_player_links(connection)
            connection.execute("DROP VIEW IF EXISTS unified_match_stream")
            connection.execute(UNIFIED_MATCH_STREAM_VIEW_SQL)
            connection.execute("DROP VIEW IF EXISTS player_latest_utr")
            connection.execute(
                """
                CREATE VIEW player_latest_utr AS
                SELECT
                    p.player_id,
                    p.full_name,
                    p.full_name_normalized,
                    p.latest_utr_singles,
                    p.latest_utr_rating_date,
                    p.latest_utr_rank,
                    p.latest_utr_three_month_rating,
                    p.latest_utr_nationality,
                    p.latest_utr_provider_player_id
                FROM players p
                WHERE p.latest_utr_singles IS NOT NULL
                """
            )
            connection.execute("DROP VIEW IF EXISTS player_profile_view")
            connection.execute(
                """
                CREATE VIEW player_profile_view AS
                WITH latest_rankings AS (
                    SELECT
                        r.player_id,
                        r.ranking_date,
                        r.rank,
                        r.points
                    FROM player_rankings r
                    JOIN (
                        SELECT
                            player_id,
                            MAX(ranking_date) AS ranking_date
                        FROM player_rankings
                        GROUP BY player_id
                    ) latest
                      ON latest.player_id = r.player_id
                     AND latest.ranking_date = r.ranking_date
                ),
                career_match_stats AS (
                    SELECT
                        player_id,
                        COUNT(*) AS total_matches,
                        SUM(is_winner) AS total_wins,
                        COUNT(*) - SUM(is_winner) AS total_losses,
                        AVG(minutes) AS avg_match_minutes,
                        SUM(COALESCE(ace, 0)) AS total_aces,
                        SUM(COALESCE(df, 0)) AS total_double_faults,
                        MAX(match_id) AS latest_stats_match_id
                    FROM player_match_stats
                    GROUP BY player_id
                ),
                latest_load AS (
                    SELECT
                        l.player_id,
                        l.match_id,
                        l.match_start_time_utc,
                        l.player_minutes_last_7d,
                        l.player_minutes_last_14d,
                        l.player_matches_last_7d,
                        l.opponent_minutes_last_7d,
                        l.minutes_diff_last_7d,
                        l.travel_fatigue_score,
                        l.load_score,
                        l.load_plus_travel_score,
                        l.tz_shift_abs,
                        l.days_rest
                    FROM player_match_load_features l
                    JOIN (
                        SELECT
                            player_id,
                            MAX(match_start_time_utc) AS match_start_time_utc
                        FROM player_match_load_features
                        GROUP BY player_id
                    ) latest
                      ON latest.player_id = l.player_id
                     AND latest.match_start_time_utc = l.match_start_time_utc
                ),
                latest_match_results AS (
                    SELECT
                        s.player_id,
                        s.match_id,
                        s.opponent_id,
                        s.is_winner,
                        s.minutes,
                        m.start_time_utc,
                        m.tournament,
                        m.round
                    FROM player_match_stats s
                    JOIN matches m
                      ON m.match_id = s.match_id
                    JOIN (
                        SELECT
                            s2.player_id,
                            MAX(m2.start_time_utc) AS start_time_utc
                        FROM player_match_stats s2
                        JOIN matches m2
                          ON m2.match_id = s2.match_id
                        GROUP BY s2.player_id
                    ) latest
                      ON latest.player_id = s.player_id
                     AND latest.start_time_utc = m.start_time_utc
                )
                SELECT
                    p.player_id,
                    p.full_name,
                    p.full_name_normalized,
                    p.ioc,
                    p.hand,
                    p.dob,
                    p.height,
                    lr.ranking_date AS latest_atp_ranking_date,
                    lr.rank AS latest_atp_rank,
                    lr.points AS latest_atp_rank_points,
                    p.latest_utr_rating_date,
                    p.latest_utr_singles,
                    p.latest_utr_rank,
                    p.latest_utr_three_month_rating,
                    p.latest_utr_nationality,
                    p.latest_utr_provider_player_id,
                    cms.total_matches,
                    cms.total_wins,
                    cms.total_losses,
                    cms.avg_match_minutes,
                    cms.total_aces,
                    cms.total_double_faults,
                    ll.match_id AS latest_load_match_id,
                    ll.match_start_time_utc AS latest_load_match_time_utc,
                    ll.player_minutes_last_7d,
                    ll.player_minutes_last_14d,
                    ll.player_matches_last_7d,
                    ll.opponent_minutes_last_7d,
                    ll.minutes_diff_last_7d,
                    ll.travel_fatigue_score,
                    ll.load_score,
                    ll.load_plus_travel_score,
                    ll.tz_shift_abs AS latest_tz_shift_abs,
                    ll.days_rest AS latest_days_rest,
                    lmr.match_id AS latest_match_id,
                    lmr.start_time_utc AS latest_match_time_utc,
                    lmr.tournament AS latest_match_tournament,
                    lmr.round AS latest_match_round,
                    lmr.opponent_id AS latest_opponent_id,
                    opp.full_name AS latest_opponent_name,
                    lmr.is_winner AS latest_match_is_winner,
                    lmr.minutes AS latest_match_minutes
                FROM players p
                LEFT JOIN latest_rankings lr
                  ON lr.player_id = p.player_id
                LEFT JOIN career_match_stats cms
                  ON cms.player_id = p.player_id
                LEFT JOIN latest_load ll
                  ON ll.player_id = p.player_id
                LEFT JOIN latest_match_results lmr
                  ON lmr.player_id = p.player_id
                LEFT JOIN players opp
                  ON opp.player_id = lmr.opponent_id
                """
            )
            connection.execute("DROP VIEW IF EXISTS player_timeline_view")
            connection.execute(
                """
                CREATE VIEW player_timeline_view AS
                SELECT
                    p.player_id,
                    p.full_name,
                    r.ranking_date AS event_time_utc,
                    'atp_ranking' AS event_type,
                    NULL AS match_id,
                    NULL AS opponent_id,
                    NULL AS opponent_name,
                    NULL AS tournament,
                    NULL AS round,
                    NULL AS result_text,
                    r.rank AS atp_rank,
                    r.points AS atp_rank_points,
                    NULL AS utr_singles,
                    NULL AS utr_rank,
                    NULL AS three_month_rating,
                    NULL AS ace,
                    NULL AS df,
                    NULL AS minutes,
                    NULL AS load_score,
                    NULL AS load_plus_travel_score
                FROM player_rankings r
                JOIN players p
                  ON p.player_id = r.player_id

                UNION ALL

                SELECT
                    p.player_id,
                    p.full_name,
                    u.rating_date AS event_time_utc,
                    'utr_rating' AS event_type,
                    NULL AS match_id,
                    NULL AS opponent_id,
                    NULL AS opponent_name,
                    NULL AS tournament,
                    NULL AS round,
                    NULL AS result_text,
                    NULL AS atp_rank,
                    NULL AS atp_rank_points,
                    u.utr_singles,
                    u.utr_rank,
                    u.three_month_rating,
                    NULL AS ace,
                    NULL AS df,
                    NULL AS minutes,
                    NULL AS load_score,
                    NULL AS load_plus_travel_score
                FROM player_utr_ratings u
                JOIN players p
                  ON p.player_id = u.canonical_player_id

                UNION ALL

                SELECT
                    p.player_id,
                    p.full_name,
                    m.start_time_utc AS event_time_utc,
                    'match' AS event_type,
                    s.match_id,
                    s.opponent_id,
                    opp.full_name AS opponent_name,
                    m.tournament,
                    m.round,
                    CASE WHEN s.is_winner = 1 THEN 'win' ELSE 'loss' END AS result_text,
                    NULL AS atp_rank,
                    NULL AS atp_rank_points,
                    NULL AS utr_singles,
                    NULL AS utr_rank,
                    NULL AS three_month_rating,
                    s.ace,
                    s.df,
                    s.minutes,
                    l.load_score,
                    l.load_plus_travel_score
                FROM player_match_stats s
                JOIN players p
                  ON p.player_id = s.player_id
                LEFT JOIN players opp
                  ON opp.player_id = s.opponent_id
                LEFT JOIN matches m
                  ON m.match_id = s.match_id
                LEFT JOIN player_match_load_features l
                  ON l.match_id = s.match_id
                 AND l.player_id = s.player_id
                """
            )

    def _ensure_column(
        self,
        connection: ConnectionWrapper,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})")
        }
        if column_name in columns:
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _temporal_column_specs(self) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
        return [
            ("players", ("dob", "latest_utr_rating_date"), ("updated_at",)),
            ("player_utr_ratings", ("rating_date",), ("updated_at",)),
            ("player_rankings", ("ranking_date",), ("updated_at",)),
            ("historical_matches", ("tourney_date",), ("ingested_at",)),
            ("matches", (), ("start_time_utc", "end_time_utc", "updated_at")),
            (
                "player_match_load_features",
                (),
                ("match_start_time_utc", "prev_match_end_utc", "current_match_start_utc", "updated_at"),
            ),
            ("live_match_feed", ("match_date",), ("first_seen_at", "last_seen_at")),
            ("live_match_snapshots", (), ("captured_at",)),
            ("flashscore_match_feed", (), ("first_seen_at", "last_seen_at")),
            ("flashscore_point_snapshots", (), ("captured_at",)),
            ("grand_slam_match_feed", (), ("first_seen_at", "last_seen_at")),
            ("grand_slam_match_snapshots", (), ("captured_at",)),
            ("ingestion_runs", (), ("started_at", "completed_at")),
            ("source_files", (), ("synced_at",)),
            ("tournaments", (), ("updated_at",)),
            ("tournament_editions", ("start_date", "end_date"), ("updated_at",)),
            ("tournament_locations", (), ("updated_at",)),
        ]

    def _get_metadata_value(self, connection: ConnectionWrapper, meta_key: str) -> str | None:
        row = connection.execute(
            "SELECT meta_value FROM pipeline_metadata WHERE meta_key = ?",
            (meta_key,),
        ).fetchone()
        if row is None:
            return None
        return _clean_text(row["meta_value"])

    def _set_metadata_value(self, connection: ConnectionWrapper, meta_key: str, meta_value: str) -> None:
        connection.execute(
            _upsert_sql("pipeline_metadata", ("meta_key", "meta_value", "updated_at"), ("meta_key",)),
            (meta_key, meta_value, _utc_now()),
        )

    def _maybe_normalize_temporal_columns(self, connection: ConnectionWrapper) -> None:
        normalization_key = "temporal_columns_normalized_v1"
        if self._get_metadata_value(connection, normalization_key) == "1":
            return

        temporal_columns = self._temporal_column_specs()
        has_any_rows = False
        for table_name, _, _ in temporal_columns:
            if connection.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone() is not None:
                has_any_rows = True
                break
        if not has_any_rows:
            self._set_metadata_value(connection, normalization_key, "1")
            return

        for table_name, date_columns, datetime_columns in temporal_columns:
            self._normalize_table_temporal_columns(
                connection,
                table_name,
                date_columns=date_columns,
                datetime_columns=datetime_columns,
            )
        self._set_metadata_value(connection, normalization_key, "1")

    def _normalize_table_temporal_columns(
        self,
        connection: ConnectionWrapper,
        table_name: str,
        *,
        date_columns: Sequence[str],
        datetime_columns: Sequence[str],
    ) -> None:
        existing_columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})")
        }
        active_date_columns = [column for column in date_columns if column in existing_columns]
        active_datetime_columns = [column for column in datetime_columns if column in existing_columns]
        selected_columns = active_date_columns + active_datetime_columns
        if not selected_columns:
            return

        rows = connection.execute(
            f'SELECT rowid, {", ".join(selected_columns)} FROM {table_name}'
        ).fetchall()
        if not rows:
            return

        update_sql = f'UPDATE {table_name} SET {", ".join(f"{column} = ?" for column in selected_columns)} WHERE rowid = ?'
        pending_updates: list[tuple[Any, ...]] = []
        for row in rows:
            new_values: list[Any] = []
            changed = False
            for column in active_date_columns:
                normalized = _clean_date(row[column])
                if normalized != row[column]:
                    changed = True
                new_values.append(normalized)
            for column in active_datetime_columns:
                normalized = _clean_utc_datetime(row[column])
                if normalized != row[column]:
                    changed = True
                new_values.append(normalized)
            if changed:
                pending_updates.append((*new_values, row["rowid"]))
        if pending_updates:
            connection.executemany(update_sql, pending_updates)

    def start_run(self, pipeline_name: str, details: dict[str, Any] | None = None) -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs (run_id, pipeline_name, started_at, status, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, pipeline_name, _utc_now(), "running", _json_dumps(details or {})),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, details: dict[str, Any] | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                SET completed_at = ?, status = ?, details_json = ?
                WHERE run_id = ?
                """,
                (_utc_now(), status, _json_dumps(details or {}), run_id),
            )

    def _source_file_changed(
        self,
        connection: ConnectionWrapper,
        source_path: Path,
        *,
        source_type: str,
        force: bool,
    ) -> bool:
        if force:
            return True
        stat = source_path.stat()
        existing = connection.execute(
            """
            SELECT file_size, modified_time_ns
            FROM source_files
            WHERE source_path = ?
            """,
            (str(source_path),),
        ).fetchone()
        if existing is None:
            return True
        return bool(
            existing["file_size"] != stat.st_size
            or existing["modified_time_ns"] != stat.st_mtime_ns
        )

    def _record_source_file(
        self,
        connection: ConnectionWrapper,
        source_path: Path,
        *,
        source_type: str,
        row_count: int,
    ) -> None:
        stat = source_path.stat()
        connection.execute(
            """
            INSERT INTO source_files (
                source_path, source_type, file_size, modified_time_ns, synced_at, row_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_path) DO UPDATE SET
                source_type = excluded.source_type,
                file_size = excluded.file_size,
                modified_time_ns = excluded.modified_time_ns,
                synced_at = excluded.synced_at,
                row_count = excluded.row_count
            """,
            (str(source_path), source_type, stat.st_size, stat.st_mtime_ns, _utc_now(), row_count),
        )

    def sync_players_file(self, player_csv: str | Path, *, force: bool = False) -> SyncSummary:
        path = Path(player_csv)
        if not path.exists():
            raise FileNotFoundError(f"Player CSV not found: {path}")

        summary = SyncSummary()
        upsert_sql = _upsert_sql("players", PLAYER_COLUMNS, ["player_id"])

        with self.connect() as connection:
            if not self._source_file_changed(connection, path, source_type="players_csv", force=force):
                summary.skipped_files = 1
                return summary

            now = _utc_now()
            rows_read = 0
            existing_ids = {
                row["player_id"]
                for row in connection.execute("SELECT player_id FROM players")
            }

            def iter_rows() -> Iterable[tuple[Any, ...]]:
                nonlocal rows_read
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        rows_read += 1
                        full_name = _clean_text(
                            " ".join(
                                part
                                for part in [
                                    _clean_text(row.get("name_first")),
                                    _clean_text(row.get("name_last")),
                                ]
                                if part
                            )
                        )
                        yield (
                            _clean_identifier(row.get("player_id")),
                            full_name,
                            _clean_text(row.get("name_first")),
                            _clean_text(row.get("name_last")),
                            normalize_player_name(full_name or ""),
                            _clean_text(row.get("hand")),
                            _clean_date(row.get("dob"), input_format="%Y%m%d"),
                            _clean_text(row.get("ioc")),
                            _clean_int(row.get("height")),
                            _clean_text(row.get("wikidata_id")),
                            str(path),
                            now,
                        )

            for batch in _batched(iter_rows()):
                for record in batch:
                    player_id = record[0]
                    if player_id in existing_ids:
                        summary.updated += 1
                    else:
                        summary.inserted += 1
                        existing_ids.add(player_id)
                connection.executemany(upsert_sql, batch)

            summary.rows_read = rows_read
            summary.synced_files = 1
            self._record_source_file(connection, path, source_type="players_csv", row_count=rows_read)
        return summary

    def players_for_utr_site_refresh(
        self,
        *,
        active_since: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if active_since:
            query = """
                SELECT
                    p.player_id,
                    p.full_name,
                    p.ioc
                FROM players p
                WHERE p.full_name IS NOT NULL
                  AND EXISTS (
                    SELECT 1
                    FROM matches m
                    WHERE (m.winner_player_id = p.player_id OR m.loser_player_id = p.player_id)
                      AND m.start_time_utc >= ?
                  )
                ORDER BY p.full_name, p.player_id
            """
            parameters: list[Any] = [active_since]
        else:
            query = """
                SELECT
                    p.player_id,
                    p.full_name,
                    p.ioc
                FROM players p
                WHERE p.full_name IS NOT NULL
                ORDER BY p.full_name, p.player_id
            """
            parameters = []
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]

    def sync_player_aliases(
        self,
        *,
        alias_csv: str | Path | None = None,
    ) -> SyncSummary:
        summary = SyncSummary()
        upsert_sql = _upsert_sql(
            "player_aliases",
            PLAYER_ALIAS_COLUMNS,
            ["alias_lookup", "canonical_player_id", "source_system"],
        )
        with self.connect() as connection:
            players = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT
                        player_id,
                        full_name,
                        full_name_normalized
                    FROM players
                    WHERE full_name IS NOT NULL
                    """
                ).fetchall()
            ]
            players_by_lookup: dict[str, list[dict[str, Any]]] = {}
            for player in players:
                lookup = _clean_text(player.get("full_name_normalized")) or normalize_player_name(
                    player.get("full_name") or ""
                )
                if lookup:
                    players_by_lookup.setdefault(lookup, []).append(player)

            existing_keys = {
                (row["alias_lookup"], row["canonical_player_id"], row["source_system"])
                for row in connection.execute(
                    "SELECT alias_lookup, canonical_player_id, source_system FROM player_aliases"
                )
            }
            now = _utc_now()
            records: list[tuple[Any, ...]] = []

            def add_alias_record(
                *,
                alias_name: str | None,
                canonical_player_id: str | None,
                canonical_player_name: str | None,
                source_system: str,
                confidence_score: float,
                source_file: str,
            ) -> None:
                cleaned_alias_name = _clean_text(alias_name)
                cleaned_player_id = _clean_text(canonical_player_id)
                cleaned_player_name = _clean_text(canonical_player_name)
                alias_lookup = normalize_player_name(cleaned_alias_name or "")
                if (
                    cleaned_alias_name is None
                    or cleaned_player_id is None
                    or cleaned_player_name is None
                    or not alias_lookup
                ):
                    return
                key = (alias_lookup, cleaned_player_id, source_system)
                if key in existing_keys:
                    summary.updated += 1
                else:
                    summary.inserted += 1
                    existing_keys.add(key)
                records.append(
                    (
                        alias_lookup,
                        cleaned_alias_name,
                        cleaned_player_id,
                        cleaned_player_name,
                        source_system,
                        float(confidence_score),
                        source_file,
                        now,
                    )
                )

            for player in players:
                add_alias_record(
                    alias_name=player.get("full_name"),
                    canonical_player_id=player.get("player_id"),
                    canonical_player_name=player.get("full_name"),
                    source_system="players",
                    confidence_score=1.0,
                    source_file="players",
                )

            utr_alias_rows = connection.execute(
                """
                SELECT DISTINCT
                    player_name,
                    canonical_player_name,
                    source_file
                FROM player_utr_ratings
                WHERE player_name IS NOT NULL
                  AND canonical_player_name IS NOT NULL
                  AND player_name <> canonical_player_name
                """
            ).fetchall()
            for row in utr_alias_rows:
                canonical_lookup = normalize_player_name(row["canonical_player_name"] or "")
                for player in players_by_lookup.get(canonical_lookup, []):
                    add_alias_record(
                        alias_name=row["player_name"],
                        canonical_player_id=player.get("player_id"),
                        canonical_player_name=player.get("full_name"),
                        source_system="utr_ratings",
                        confidence_score=0.95,
                        source_file=_clean_text(row["source_file"]) or "player_utr_ratings",
                    )

            if alias_csv is not None:
                alias_path = Path(alias_csv)
                if not alias_path.exists():
                    raise FileNotFoundError(f"Alias CSV not found: {alias_path}")
                rows_read = 0
                with alias_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        rows_read += 1
                        source_name = _clean_text(row.get("source_name"))
                        canonical_name = _clean_text(row.get("canonical_name"))
                        if source_name is None or canonical_name is None:
                            continue
                        canonical_lookup = normalize_player_name(canonical_name)
                        for player in players_by_lookup.get(canonical_lookup, []):
                            add_alias_record(
                                alias_name=source_name,
                                canonical_player_id=player.get("player_id"),
                                canonical_player_name=player.get("full_name"),
                                source_system="utr_alias_csv",
                                confidence_score=1.0,
                                source_file=str(alias_path),
                            )
                summary.rows_read += rows_read
                summary.synced_files += 1
                self._record_source_file(
                    connection,
                    alias_path,
                    source_type="player_alias_csv",
                    row_count=rows_read,
                )

            if records:
                for batch in _batched(records):
                    connection.executemany(upsert_sql, batch)
            self._refresh_utr_player_links(connection)
            self._refresh_latest_utr_for_players(connection)
            if alias_csv is None:
                summary.synced_files = 1
            summary.rows_read += len(players) + len(utr_alias_rows)
        return summary

    def sync_ranking_files(
        self,
        data_dir: str | Path,
        *,
        ranking_files: Sequence[str] | None = None,
        force: bool = False,
    ) -> SyncSummary:
        base_dir = Path(data_dir)
        files = _resolve_ranking_files(base_dir, ranking_files)
        summary = SyncSummary()
        upsert_sql = _upsert_sql("player_rankings", PLAYER_RANKING_COLUMNS, ["ranking_date", "player_id"])

        with self.connect() as connection:
            now = _utc_now()

            for path in files:
                if not self._source_file_changed(connection, path, source_type="rankings_csv", force=force):
                    summary.skipped_files += 1
                    continue

                before_count = int(
                    connection.execute("SELECT COUNT(*) AS count FROM player_rankings").fetchone()["count"]
                )
                rows_read = 0
                batch_records: list[tuple[Any, ...]] = []
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        rows_read += 1
                        record = (
                            _clean_date(row.get("ranking_date"), input_format="%Y%m%d"),
                            _clean_identifier(row.get("player")),
                            _clean_int(row.get("rank")),
                            _clean_int(row.get("points")),
                            str(path),
                            now,
                        )
                        if record[0] is None or record[1] is None:
                            continue
                        batch_records.append(record)
                        if len(batch_records) >= 2000:
                            connection.executemany(upsert_sql, batch_records)
                            batch_records = []
                if batch_records:
                    connection.executemany(upsert_sql, batch_records)

                after_count = int(
                    connection.execute("SELECT COUNT(*) AS count FROM player_rankings").fetchone()["count"]
                )
                inserted = max(after_count - before_count, 0)
                summary.inserted += inserted
                summary.updated += max(rows_read - inserted, 0)

                summary.rows_read += rows_read
                summary.synced_files += 1
                self._record_source_file(connection, path, source_type="rankings_csv", row_count=rows_read)

        return summary

    def sync_utr_file(
        self,
        utr_csv: str | Path,
        *,
        alias_csv: str | Path | None = None,
        default_rating_date: str | None = None,
        force: bool = False,
        source_type: str = "utr_history_csv",
    ) -> SyncSummary:
        path = Path(utr_csv)
        if not path.exists():
            raise FileNotFoundError(f"UTR CSV not found: {path}")

        aliases = load_name_aliases(alias_csv)
        summary = SyncSummary()
        upsert_sql = _upsert_sql("player_utr_ratings", PLAYER_UTR_COLUMNS, ["player_lookup", "rating_date"])

        with self.connect() as connection:
            if not self._source_file_changed(connection, path, source_type=source_type, force=force):
                summary.skipped_files = 1
                return summary

            header = self._csv_columns(path)
            required_columns = {"player_name", "utr_singles"}
            missing = required_columns - header
            if missing:
                raise ValueError(f"UTR CSV is missing required columns: {sorted(missing)}")
            if "rating_date" not in header and default_rating_date is None:
                raise ValueError(
                    "UTR CSV is missing rating_date. Provide a rating_date column or pass default_rating_date."
                )

            before_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM player_utr_ratings").fetchone()["count"]
            )
            now = _utc_now()
            rows_read = 0
            upserted_rows = 0
            batch_records: list[tuple[Any, ...]] = []

            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows_read += 1
                    source_name = _clean_text(row.get("player_name"))
                    utr_value = _clean_float(row.get("utr_singles"))
                    rating_date = _clean_date(row.get("rating_date")) if "rating_date" in row else None
                    if rating_date is None and default_rating_date is not None:
                        rating_date = _clean_date(default_rating_date)
                    if not source_name or utr_value is None or rating_date is None:
                        continue

                    canonical_name = aliases.get(normalize_player_name(source_name), source_name)
                    player_lookup = normalize_player_name(canonical_name)
                    batch_records.append(
                        (
                            player_lookup,
                            source_name,
                            canonical_name,
                            None,
                            rating_date,
                            utr_value,
                            _clean_int(row.get("utr_rank")),
                            _clean_float(row.get("three_month_rating")),
                            _clean_text(row.get("nationality")),
                            _clean_text(row.get("provider_player_id")),
                            str(path),
                            source_type,
                            now,
                        )
                    )
                    upserted_rows += 1
                    if len(batch_records) >= 1000:
                        connection.executemany(upsert_sql, batch_records)
                        batch_records = []
            if batch_records:
                connection.executemany(upsert_sql, batch_records)

            after_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM player_utr_ratings").fetchone()["count"]
            )
            inserted = max(after_count - before_count, 0)
            summary.inserted += inserted
            summary.updated += max(upserted_rows - inserted, 0)
            summary.rows_read = rows_read
            summary.synced_files = 1
            self._refresh_utr_player_links(connection)
            self._refresh_latest_utr_for_players(connection)
            self._record_source_file(connection, path, source_type=source_type, row_count=rows_read)
        return summary

    def sync_tournament_locations_file(
        self,
        tournament_locations_csv: str | Path,
        *,
        force: bool = False,
    ) -> SyncSummary:
        path = Path(tournament_locations_csv)
        if not path.exists():
            raise FileNotFoundError(f"Tournament locations CSV not found: {path}")

        summary = SyncSummary()
        upsert_sql = _upsert_sql("tournament_locations", TOURNAMENT_LOCATION_COLUMNS, ["tournament_lookup"])
        with self.connect() as connection:
            if not self._source_file_changed(connection, path, source_type="tournament_locations_csv", force=force):
                summary.skipped_files = 1
                return summary
            now = _utc_now()
            rows_read = 0
            existing_keys = {
                row["tournament_lookup"]
                for row in connection.execute("SELECT tournament_lookup FROM tournament_locations")
            }
            batch_records: list[tuple[Any, ...]] = []
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows_read += 1
                    tournament_name = _clean_text(row.get("tournament_name"))
                    tournament_lookup = _normalize_tournament_lookup(
                        row.get("tournament_lookup") or tournament_name
                    )
                    if tournament_lookup is None or tournament_name is None:
                        continue
                    if tournament_lookup in existing_keys:
                        summary.updated += 1
                    else:
                        summary.inserted += 1
                        existing_keys.add(tournament_lookup)
                    batch_records.append(
                        (
                            tournament_lookup,
                            tournament_name,
                            _clean_text(row.get("city")),
                            _clean_text(row.get("country")),
                            _clean_text(row.get("timezone_name")),
                            _clean_float(row.get("utc_offset_hours")),
                            _clean_float(row.get("latitude")),
                            _clean_float(row.get("longitude")),
                            str(path),
                            now,
                        )
                    )
            if batch_records:
                connection.executemany(upsert_sql, batch_records)
            summary.rows_read = rows_read
            summary.synced_files = 1
            self._record_source_file(connection, path, source_type="tournament_locations_csv", row_count=rows_read)
        return summary

    def _csv_columns(self, path: Path) -> set[str]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return set()
        return {str(column).strip() for column in header if str(column).strip()}

    def _refresh_latest_utr_for_players(self, connection: ConnectionWrapper) -> None:
        connection.execute(
            """
            UPDATE players
            SET latest_utr_singles = (
                    SELECT u.utr_singles
                    FROM player_utr_ratings u
                    WHERE u.canonical_player_id = players.player_id
                       OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
                    ORDER BY
                        CASE WHEN u.canonical_player_id = players.player_id THEN 0 ELSE 1 END,
                        u.rating_date DESC
                    LIMIT 1
                ),
                latest_utr_rating_date = (
                    SELECT u.rating_date
                    FROM player_utr_ratings u
                    WHERE u.canonical_player_id = players.player_id
                       OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
                    ORDER BY
                        CASE WHEN u.canonical_player_id = players.player_id THEN 0 ELSE 1 END,
                        u.rating_date DESC
                    LIMIT 1
                ),
                latest_utr_rank = (
                    SELECT u.utr_rank
                    FROM player_utr_ratings u
                    WHERE u.canonical_player_id = players.player_id
                       OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
                    ORDER BY
                        CASE WHEN u.canonical_player_id = players.player_id THEN 0 ELSE 1 END,
                        u.rating_date DESC
                    LIMIT 1
                ),
                latest_utr_three_month_rating = (
                    SELECT u.three_month_rating
                    FROM player_utr_ratings u
                    WHERE u.canonical_player_id = players.player_id
                       OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
                    ORDER BY
                        CASE WHEN u.canonical_player_id = players.player_id THEN 0 ELSE 1 END,
                        u.rating_date DESC
                    LIMIT 1
                ),
                latest_utr_nationality = (
                    SELECT u.nationality
                    FROM player_utr_ratings u
                    WHERE u.canonical_player_id = players.player_id
                       OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
                    ORDER BY
                        CASE WHEN u.canonical_player_id = players.player_id THEN 0 ELSE 1 END,
                        u.rating_date DESC
                    LIMIT 1
                ),
                latest_utr_provider_player_id = (
                    SELECT u.provider_player_id
                    FROM player_utr_ratings u
                    WHERE u.canonical_player_id = players.player_id
                       OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
                    ORDER BY
                        CASE WHEN u.canonical_player_id = players.player_id THEN 0 ELSE 1 END,
                        u.rating_date DESC
                    LIMIT 1
                )
            WHERE EXISTS (
                SELECT 1
                FROM player_utr_ratings u
                WHERE u.canonical_player_id = players.player_id
                   OR (u.canonical_player_id IS NULL AND u.player_lookup = players.full_name_normalized)
            )
            """
        )

    def _refresh_utr_player_links(self, connection: ConnectionWrapper) -> None:
        connection.execute(
            """
            UPDATE player_utr_ratings
            SET canonical_player_id = COALESCE(
                    (
                        SELECT p.player_id
                        FROM players p
                        WHERE p.full_name_normalized = player_utr_ratings.player_lookup
                        LIMIT 1
                    ),
                    (
                        SELECT pa.canonical_player_id
                        FROM player_aliases pa
                        WHERE pa.alias_lookup = player_utr_ratings.player_lookup
                        ORDER BY pa.confidence_score DESC, pa.updated_at DESC
                        LIMIT 1
                    )
                )
            """
        )

    def sync_historical_matches(
        self,
        data_dir: str | Path,
        *,
        years: Sequence[str] | None = None,
        include_ongoing: bool = False,
        force: bool = False,
    ) -> SyncSummary:
        base_dir = Path(data_dir)
        files = _resolve_main_tour_files(base_dir, years, include_ongoing)
        summary = SyncSummary()
        upsert_sql = _upsert_sql("historical_matches", HISTORICAL_MATCH_COLUMNS, ["match_key"])

        with self.connect() as connection:
            now = _utc_now()

            for path in files:
                if not self._source_file_changed(connection, path, source_type="main_tour_matches_csv", force=force):
                    summary.skipped_files += 1
                    continue

                before_count = int(
                    connection.execute("SELECT COUNT(*) AS count FROM historical_matches").fetchone()["count"]
                )
                rows_read = 0
                batch_records: list[tuple[Any, ...]] = []
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        rows_read += 1
                        record = (
                            _historical_match_key(row),
                            str(path),
                            _source_year_from_path(path),
                            _clean_identifier(row.get("tourney_id")),
                            _clean_text(row.get("tourney_name")),
                            _clean_text(row.get("surface")),
                            _clean_text(row.get("indoor")),
                            _clean_int(row.get("draw_size")),
                            _clean_text(row.get("tourney_level")),
                            _clean_date(row.get("tourney_date"), input_format="%Y%m%d"),
                            _clean_int(row.get("match_num")),
                            _clean_identifier(row.get("winner_id")),
                            _clean_text(row.get("winner_seed")),
                            _clean_text(row.get("winner_entry")),
                            _clean_text(row.get("winner_name")),
                            _clean_text(row.get("winner_hand")),
                            _clean_int(row.get("winner_ht")),
                            _clean_text(row.get("winner_ioc")),
                            _clean_float(row.get("winner_age")),
                            _clean_int(row.get("winner_rank")),
                            _clean_int(row.get("winner_rank_points")),
                            _clean_identifier(row.get("loser_id")),
                            _clean_text(row.get("loser_seed")),
                            _clean_text(row.get("loser_entry")),
                            _clean_text(row.get("loser_name")),
                            _clean_text(row.get("loser_hand")),
                            _clean_int(row.get("loser_ht")),
                            _clean_text(row.get("loser_ioc")),
                            _clean_float(row.get("loser_age")),
                            _clean_int(row.get("loser_rank")),
                            _clean_int(row.get("loser_rank_points")),
                            _clean_text(row.get("score")),
                            _clean_int(row.get("best_of")),
                            _clean_text(row.get("round")),
                            _clean_int(row.get("minutes")),
                            _clean_int(row.get("w_ace")),
                            _clean_int(row.get("w_df")),
                            _clean_int(row.get("w_svpt")),
                            _clean_int(row.get("w_1stIn")),
                            _clean_int(row.get("w_1stWon")),
                            _clean_int(row.get("w_2ndWon")),
                            _clean_int(row.get("w_SvGms")),
                            _clean_int(row.get("w_bpSaved")),
                            _clean_int(row.get("w_bpFaced")),
                            _clean_int(row.get("l_ace")),
                            _clean_int(row.get("l_df")),
                            _clean_int(row.get("l_svpt")),
                            _clean_int(row.get("l_1stIn")),
                            _clean_int(row.get("l_1stWon")),
                            _clean_int(row.get("l_2ndWon")),
                            _clean_int(row.get("l_SvGms")),
                            _clean_int(row.get("l_bpSaved")),
                            _clean_int(row.get("l_bpFaced")),
                            now,
                        )
                        batch_records.append(record)
                        if len(batch_records) >= 1000:
                            connection.executemany(upsert_sql, batch_records)
                            batch_records = []
                if batch_records:
                    connection.executemany(upsert_sql, batch_records)

                after_count = int(
                    connection.execute("SELECT COUNT(*) AS count FROM historical_matches").fetchone()["count"]
                )
                inserted = max(after_count - before_count, 0)
                summary.inserted += inserted
                summary.updated += max(rows_read - inserted, 0)

                summary.rows_read += rows_read
                summary.synced_files += 1
                self._record_source_file(connection, path, source_type="main_tour_matches_csv", row_count=rows_read)
        return summary

    def sync_matches_table(self) -> SyncSummary:
        summary = SyncSummary()
        upsert_sql = _upsert_sql("matches", MATCH_FACT_COLUMNS, ["match_id"])
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    match_key,
                    tourney_id,
                    tourney_date,
                    minutes,
                    surface,
                    best_of,
                    tourney_name,
                    round_name,
                    score,
                    winner_id,
                    loser_id,
                    winner_name,
                    loser_name,
                    tourney_level
                FROM historical_matches
                ORDER BY tourney_date, match_num, match_key
                """
            ).fetchall()
            summary.rows_read = len(rows)
            summary.synced_files = len(rows)
            existing_ids = {
                row["match_id"]
                for row in connection.execute("SELECT match_id FROM matches WHERE source_system = 'historical_main_tour'")
            }
            now = _utc_now()
            records: list[tuple[Any, ...]] = []
            for row in rows:
                match_id = _clean_text(row["match_key"])
                start_time_utc = _historical_match_start_time(row["tourney_date"])
                if match_id is None or start_time_utc is None:
                    continue
                if match_id in existing_ids:
                    summary.updated += 1
                else:
                    summary.inserted += 1
                records.append(
                    (
                        match_id,
                        match_id,
                        "historical_main_tour",
                        _tournament_id(row["tourney_id"], row["tourney_name"], row["tourney_level"]),
                        _tournament_edition_id(
                            row["tourney_id"],
                            row["tourney_name"],
                            row["tourney_level"],
                            row["tourney_date"],
                        ),
                        start_time_utc,
                        _match_end_time(start_time_utc, row["minutes"]),
                        _clean_int(row["minutes"]),
                        _clean_text(row["surface"]),
                        _clean_int(row["best_of"]),
                        _clean_text(row["tourney_name"]),
                        _clean_text(row["round_name"]),
                        _clean_text(row["score"]),
                        _player_reference(row["winner_id"], row["winner_name"]),
                        _player_reference(row["loser_id"], row["loser_name"]),
                        _clean_text(row["winner_name"]),
                        _clean_text(row["loser_name"]),
                        _clean_text(row["tourney_level"]),
                        now,
                    )
                )
            for batch in _batched(records):
                connection.executemany(upsert_sql, batch)
        return summary

    def sync_player_match_stats(self) -> SyncSummary:
        summary = SyncSummary()
        upsert_sql = _upsert_sql("player_match_stats", PLAYER_MATCH_STATS_COLUMNS, ["match_id", "player_id"])
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    match_key,
                    winner_id,
                    winner_name,
                    loser_id,
                    loser_name,
                    minutes,
                    w_ace,
                    w_df,
                    w_svpt,
                    w_1stIn,
                    w_1stWon,
                    w_2ndWon,
                    w_SvGms,
                    w_bpSaved,
                    w_bpFaced,
                    l_ace,
                    l_df,
                    l_svpt,
                    l_1stIn,
                    l_1stWon,
                    l_2ndWon,
                    l_SvGms,
                    l_bpSaved,
                    l_bpFaced
                FROM historical_matches
                ORDER BY tourney_date, match_num, match_key
                """
            ).fetchall()
            summary.rows_read = len(rows) * 2
            summary.synced_files = len(rows)
            existing_keys = {
                (row["match_id"], row["player_id"])
                for row in connection.execute("SELECT match_id, player_id FROM player_match_stats")
            }
            now = _utc_now()
            records: list[tuple[Any, ...]] = []
            for row in rows:
                match_id = _clean_text(row["match_key"])
                winner_id = _player_reference(row["winner_id"], row["winner_name"])
                loser_id = _player_reference(row["loser_id"], row["loser_name"])
                if match_id is None or winner_id is None or loser_id is None:
                    continue
                stat_rows = [
                    (
                        match_id,
                        winner_id,
                        loser_id,
                        1,
                        "winner",
                        _clean_int(row["w_ace"]),
                        _clean_int(row["w_df"]),
                        _clean_int(row["w_svpt"]),
                        _clean_int(row["w_1stIn"]),
                        _clean_int(row["w_1stWon"]),
                        _clean_int(row["w_2ndWon"]),
                        _clean_int(row["w_SvGms"]),
                        _clean_int(row["w_bpSaved"]),
                        _clean_int(row["w_bpFaced"]),
                        _clean_int(row["minutes"]),
                        now,
                    ),
                    (
                        match_id,
                        loser_id,
                        winner_id,
                        0,
                        "loser",
                        _clean_int(row["l_ace"]),
                        _clean_int(row["l_df"]),
                        _clean_int(row["l_svpt"]),
                        _clean_int(row["l_1stIn"]),
                        _clean_int(row["l_1stWon"]),
                        _clean_int(row["l_2ndWon"]),
                        _clean_int(row["l_SvGms"]),
                        _clean_int(row["l_bpSaved"]),
                        _clean_int(row["l_bpFaced"]),
                        _clean_int(row["minutes"]),
                        now,
                    ),
                ]
                for record in stat_rows:
                    key = (record[0], record[1])
                    if key in existing_keys:
                        summary.updated += 1
                    else:
                        summary.inserted += 1
                        existing_keys.add(key)
                    records.append(record)
            for batch in _batched(records):
                connection.executemany(upsert_sql, batch)
        return summary

    def sync_tournament_dimensions(self) -> SyncSummary:
        summary = SyncSummary()
        tournament_upsert_sql = _upsert_sql("tournaments", TOURNAMENT_COLUMNS, ["tournament_id"])
        edition_upsert_sql = _upsert_sql("tournament_editions", TOURNAMENT_EDITION_COLUMNS, ["edition_id"])
        with self.connect() as connection:
            location_lookup = {
                row["tournament_lookup"]: dict(row)
                for row in connection.execute("SELECT * FROM tournament_locations")
            }
            historical_rows = connection.execute(
                """
                SELECT
                    tourney_id,
                    tourney_name,
                    tourney_level,
                    surface,
                    indoor,
                    draw_size,
                    tourney_date,
                    source_file,
                    source_year
                FROM historical_matches
                ORDER BY tourney_date, match_num, match_key
                """
            ).fetchall()
            if not historical_rows:
                return summary

            tournaments: dict[str, dict[str, Any]] = {}
            editions: dict[str, dict[str, Any]] = {}
            for row in historical_rows:
                tournament_id = _tournament_id(row["tourney_id"], row["tourney_name"], row["tourney_level"])
                edition_id = _tournament_edition_id(
                    row["tourney_id"],
                    row["tourney_name"],
                    row["tourney_level"],
                    row["tourney_date"],
                )
                if tournament_id is None or edition_id is None:
                    continue
                tournament_lookup = _normalize_tournament_lookup(row["tourney_name"])
                location = location_lookup.get(tournament_lookup or "", {})
                season_year = _clean_int(row["source_year"])
                tournament_record = tournaments.setdefault(
                    tournament_id,
                    {
                        "tournament_id": tournament_id,
                        "canonical_tournament_name": _clean_text(row["tourney_name"]),
                        "tour_level": _clean_text(row["tourney_level"]),
                        "default_surface": _clean_text(row["surface"]),
                        "default_indoor": _clean_text(row["indoor"]),
                        "country": _clean_text(location.get("country")),
                        "city": _clean_text(location.get("city")),
                        "source_file": _clean_text(row["source_file"]) or "historical_matches",
                    },
                )
                if tournament_record["default_surface"] is None:
                    tournament_record["default_surface"] = _clean_text(row["surface"])
                if tournament_record["default_indoor"] is None:
                    tournament_record["default_indoor"] = _clean_text(row["indoor"])
                if tournament_record["country"] is None:
                    tournament_record["country"] = _clean_text(location.get("country"))
                if tournament_record["city"] is None:
                    tournament_record["city"] = _clean_text(location.get("city"))

                edition_record = editions.setdefault(
                    edition_id,
                    {
                        "edition_id": edition_id,
                        "tournament_id": tournament_id,
                        "season_year": season_year,
                        "source_tourney_id": _clean_identifier(row["tourney_id"]),
                        "tournament_name": _clean_text(row["tourney_name"]),
                        "start_date": _clean_date(row["tourney_date"]),
                        "end_date": None,
                        "city": _clean_text(location.get("city")),
                        "country": _clean_text(location.get("country")),
                        "timezone_name": _clean_text(location.get("timezone_name")),
                        "latitude": _clean_float(location.get("latitude")),
                        "longitude": _clean_float(location.get("longitude")),
                        "altitude_m": None,
                        "surface": _clean_text(row["surface"]),
                        "indoor": _clean_text(row["indoor"]),
                        "draw_size": _clean_int(row["draw_size"]),
                        "source_file": _clean_text(row["source_file"]) or "historical_matches",
                    },
                )
                if edition_record["surface"] is None:
                    edition_record["surface"] = _clean_text(row["surface"])
                if edition_record["indoor"] is None:
                    edition_record["indoor"] = _clean_text(row["indoor"])
                if edition_record["draw_size"] is None:
                    edition_record["draw_size"] = _clean_int(row["draw_size"])
                if edition_record["city"] is None:
                    edition_record["city"] = _clean_text(location.get("city"))
                if edition_record["country"] is None:
                    edition_record["country"] = _clean_text(location.get("country"))
                if edition_record["timezone_name"] is None:
                    edition_record["timezone_name"] = _clean_text(location.get("timezone_name"))
                if edition_record["latitude"] is None:
                    edition_record["latitude"] = _clean_float(location.get("latitude"))
                if edition_record["longitude"] is None:
                    edition_record["longitude"] = _clean_float(location.get("longitude"))

            summary.rows_read = len(historical_rows)
            summary.synced_files = len(editions)
            now = _utc_now()
            existing_tournament_ids = {
                row["tournament_id"] for row in connection.execute("SELECT tournament_id FROM tournaments")
            }
            existing_edition_ids = {
                row["edition_id"] for row in connection.execute("SELECT edition_id FROM tournament_editions")
            }
            tournament_records: list[tuple[Any, ...]] = []
            edition_records: list[tuple[Any, ...]] = []
            for record in tournaments.values():
                if record["tournament_id"] in existing_tournament_ids:
                    summary.updated += 1
                else:
                    summary.inserted += 1
                    existing_tournament_ids.add(record["tournament_id"])
                tournament_records.append(
                    (
                        record["tournament_id"],
                        record["canonical_tournament_name"],
                        record["tour_level"],
                        record["default_surface"],
                        record["default_indoor"],
                        record["country"],
                        record["city"],
                        record["source_file"],
                        now,
                    )
                )
            for record in editions.values():
                if record["edition_id"] in existing_edition_ids:
                    summary.updated += 1
                else:
                    summary.inserted += 1
                    existing_edition_ids.add(record["edition_id"])
                edition_records.append(
                    (
                        record["edition_id"],
                        record["tournament_id"],
                        record["season_year"],
                        record["source_tourney_id"],
                        record["tournament_name"],
                        record["start_date"],
                        record["end_date"],
                        record["city"],
                        record["country"],
                        record["timezone_name"],
                        record["latitude"],
                        record["longitude"],
                        record["altitude_m"],
                        record["surface"],
                        record["indoor"],
                        record["draw_size"],
                        record["source_file"],
                        now,
                    )
                )
            for batch in _batched(tournament_records):
                connection.executemany(tournament_upsert_sql, batch)
            for batch in _batched(edition_records):
                connection.executemany(edition_upsert_sql, batch)
        return summary

    def sync_player_match_load_features(
        self,
        *,
        load_weight_minutes_last_7d: float = 0.35,
        load_weight_matches_last_7d: float = 0.35,
        load_weight_travel_timezones_last_7d: float = 0.20,
        load_weight_tiebreaks_last_7d: float = 0.10,
    ) -> SyncSummary:
        summary = SyncSummary()
        with self.connect() as connection:
            location_lookup = {
                row["tournament_lookup"]: dict(row)
                for row in connection.execute("SELECT * FROM tournament_locations")
            }
            matches = connection.execute(
                """
                SELECT
                    match_id,
                    start_time_utc,
                    duration_minutes,
                    score_text,
                    tournament,
                    winner_player_id,
                    loser_player_id,
                    winner_name,
                    loser_name
                FROM matches
                ORDER BY start_time_utc, match_id
                """
            ).fetchall()
            summary.rows_read = len(matches)
            now = _utc_now()
            history_by_player: dict[str, list[dict[str, Any]]] = {}
            feature_records: list[tuple[Any, ...]] = []
            skipped_self_opponent_matches = 0

            def recent_events(player_id: str, as_of: datetime, days: int) -> list[dict[str, Any]]:
                events = history_by_player.get(player_id, [])
                window = timedelta(days=days)
                return [event for event in events if as_of - event["start_time"] <= window]

            def sum_minutes(events: Sequence[dict[str, Any]]) -> int:
                return int(sum(event["duration_minutes"] for event in events if event["duration_minutes"] is not None))

            def avg_minutes(events: Sequence[dict[str, Any]]) -> float | None:
                durations = [event["duration_minutes"] for event in events if event["duration_minutes"] is not None]
                if not durations:
                    return None
                return float(sum(durations) / len(durations))

            def back_to_back_long_matches(events: Sequence[dict[str, Any]]) -> int:
                count = 0
                for event in reversed(events):
                    if not event["long_match_flag"]:
                        break
                    count += 1
                return count

            def recent_timezone_abs(events: Sequence[dict[str, Any]]) -> float:
                return float(
                    sum(
                        float(event["tz_shift_abs"])
                        for event in events
                        if event.get("tz_shift_abs") is not None
                    )
                )

            for match in matches:
                match_id = _clean_text(match["match_id"])
                match_start = _parse_utc_datetime(match["start_time_utc"])
                if match_id is None or match_start is None:
                    continue
                duration_minutes = _clean_int(match["duration_minutes"])
                tiebreak_count = _count_tiebreaks(match["score_text"])
                tournament_name = _clean_text(match["tournament"])
                tournament_lookup = _normalize_tournament_lookup(tournament_name)
                current_location = location_lookup.get(tournament_lookup or "", {})
                current_tournament_city = _clean_text(current_location.get("city")) or _default_tournament_city(tournament_name)
                current_utc_offset_hours = _clean_float(current_location.get("utc_offset_hours"))
                current_latitude = _clean_float(current_location.get("latitude"))
                current_longitude = _clean_float(current_location.get("longitude"))
                winner_id = _player_reference(match["winner_player_id"], match["winner_name"])
                loser_id = _player_reference(match["loser_player_id"], match["loser_name"])
                if winner_id is None or loser_id is None:
                    continue
                if winner_id == loser_id:
                    skipped_self_opponent_matches += 1
                    continue

                current_match_events: dict[str, dict[str, Any]] = {}
                for player_id, opponent_id in ((winner_id, loser_id), (loser_id, winner_id)):
                    player_history = history_by_player.get(player_id, [])
                    opponent_history = history_by_player.get(opponent_id, [])
                    last_1 = player_history[-1]["duration_minutes"] if player_history else None
                    last_3 = sum_minutes(player_history[-3:])
                    last_7d_events = recent_events(player_id, match_start, 7)
                    last_14d_events = recent_events(player_id, match_start, 14)
                    last_30d_events = recent_events(player_id, match_start, 30)
                    opponent_last_7d_events = recent_events(opponent_id, match_start, 7)
                    player_minutes_last_7d = sum_minutes(last_7d_events)
                    opponent_minutes_last_7d = sum_minutes(opponent_last_7d_events)
                    player_matches_last_7d = len(last_7d_events)
                    opponent_matches_last_7d = len(opponent_last_7d_events)
                    tiebreaks_last_7d = int(sum(event["tiebreaks"] for event in last_7d_events))
                    last_event = player_history[-1] if player_history else None
                    prev_tournament_city = _clean_text(last_event["tournament_city"]) if last_event else None
                    prev_match_end_utc = last_event["end_time_utc"] if last_event else None
                    previous_utc_offset_hours = last_event["utc_offset_hours"] if last_event else None
                    previous_match_duration = last_event["duration_minutes"] if last_event else None
                    timezones_crossed_signed = None
                    if current_utc_offset_hours is not None and previous_utc_offset_hours is not None:
                        timezones_crossed_signed = current_utc_offset_hours - previous_utc_offset_hours
                    tz_shift_signed = timezones_crossed_signed
                    tz_shift_abs = abs(tz_shift_signed) if tz_shift_signed is not None else None
                    if tz_shift_signed is None:
                        travel_direction = None
                    elif tz_shift_signed > 0:
                        travel_direction = "east"
                    elif tz_shift_signed < 0:
                        travel_direction = "west"
                    else:
                        travel_direction = "same"
                    great_circle_km = None
                    previous_latitude = _clean_float(last_event["latitude"]) if last_event else None
                    previous_longitude = _clean_float(last_event["longitude"]) if last_event else None
                    if (
                        current_latitude is not None
                        and current_longitude is not None
                        and previous_latitude is not None
                        and previous_longitude is not None
                    ):
                        great_circle_km = _haversine_km(
                            previous_latitude,
                            previous_longitude,
                            current_latitude,
                            current_longitude,
                        )
                    elif last_event and last_event["tournament_lookup"] == tournament_lookup:
                        great_circle_km = 0.0
                    hours_between_matches = None
                    if prev_match_end_utc is not None:
                        prev_end_dt = _parse_utc_datetime(prev_match_end_utc)
                        if prev_end_dt is not None:
                            hours_between_matches = max(
                                0.0,
                                (match_start - prev_end_dt).total_seconds() / 3600.0,
                            )
                    days_rest = None if hours_between_matches is None else hours_between_matches / 24.0
                    back_to_back_week_flag = int(
                        bool(
                            last_event
                            and last_event["tournament_lookup"] != tournament_lookup
                            and days_rest is not None
                            and days_rest <= 7.0
                        )
                    )
                    eastward_shift_flag = int(bool(tz_shift_signed is not None and tz_shift_signed > 0))
                    short_recovery_after_travel = int(
                        bool(
                            tz_shift_abs is not None
                            and tz_shift_abs >= 2
                            and hours_between_matches is not None
                            and hours_between_matches < 72
                        )
                    )
                    travel_fatigue_score = (
                        0.4 * float(tz_shift_abs or 0.0)
                        + 0.3 * eastward_shift_flag
                        + 0.2 * math.log1p(float(great_circle_km or 0.0))
                        + 0.1 * (
                            max(0.0, 48.0 - float(hours_between_matches)) / 48.0
                            if hours_between_matches is not None
                            else 0.0
                        )
                    )
                    travel_timezones_last_7d = recent_timezone_abs(last_7d_events)
                    load_score = (
                        load_weight_minutes_last_7d * player_minutes_last_7d
                        + load_weight_matches_last_7d * player_matches_last_7d
                        + load_weight_travel_timezones_last_7d * travel_timezones_last_7d
                        + load_weight_tiebreaks_last_7d * tiebreaks_last_7d
                    )
                    load_plus_travel_score = float(load_score + travel_fatigue_score)
                    feature_records.append(
                        (
                            match_id,
                            player_id,
                            opponent_id,
                            match["start_time_utc"],
                            _clean_int(last_1),
                            last_3,
                            player_minutes_last_7d,
                            sum_minutes(last_14d_events),
                            avg_minutes(last_30d_events),
                            int(bool(player_history and player_history[-1]["long_match_flag"])),
                            back_to_back_long_matches(player_history),
                            player_matches_last_7d,
                            opponent_minutes_last_7d,
                            opponent_matches_last_7d,
                            player_minutes_last_7d - opponent_minutes_last_7d,
                            prev_tournament_city,
                            current_tournament_city,
                            prev_match_end_utc,
                            match["start_time_utc"],
                            _clean_float(previous_utc_offset_hours),
                            _clean_float(current_utc_offset_hours),
                            _clean_float(timezones_crossed_signed),
                            travel_direction,
                            _clean_float(great_circle_km),
                            _clean_float(hours_between_matches),
                            _clean_float(days_rest),
                            back_to_back_week_flag,
                            _clean_float(tz_shift_signed),
                            _clean_float(tz_shift_abs),
                            eastward_shift_flag,
                            short_recovery_after_travel,
                            _clean_int(previous_match_duration),
                            float(travel_fatigue_score),
                            travel_timezones_last_7d,
                            tiebreaks_last_7d,
                            float(load_score),
                            load_plus_travel_score,
                            float(eastward_shift_flag * player_minutes_last_7d),
                            float(eastward_shift_flag * (days_rest or 0.0)),
                            float((tz_shift_abs or 0.0) * float(previous_match_duration or 0)),
                            float(back_to_back_week_flag * float(tz_shift_abs or 0.0)),
                            float(load_weight_minutes_last_7d),
                            float(load_weight_matches_last_7d),
                            float(load_weight_travel_timezones_last_7d),
                            float(load_weight_tiebreaks_last_7d),
                            now,
                        )
                    )
                    current_match_events[player_id] = {
                        "start_time": match_start,
                        "end_time_utc": _match_end_time(match["start_time_utc"], duration_minutes),
                        "duration_minutes": duration_minutes,
                        "long_match_flag": bool(duration_minutes is not None and duration_minutes > 150),
                        "tiebreaks": tiebreak_count,
                        "tournament_lookup": tournament_lookup,
                        "tournament_city": current_tournament_city,
                        "utc_offset_hours": current_utc_offset_hours,
                        "latitude": current_latitude,
                        "longitude": current_longitude,
                        "tz_shift_abs": tz_shift_abs,
                    }

                history_by_player.setdefault(winner_id, []).append(current_match_events[winner_id])
                history_by_player.setdefault(loser_id, []).append(current_match_events[loser_id])

            connection.execute("DELETE FROM player_match_load_features")
            if feature_records:
                connection.executemany(
                    f"""
                    INSERT INTO player_match_load_features ({", ".join(PLAYER_MATCH_LOAD_COLUMNS)})
                    VALUES ({", ".join("?" for _ in PLAYER_MATCH_LOAD_COLUMNS)})
                    """,
                    feature_records,
                )
            summary.inserted = len(feature_records)
            summary.synced_files = len(feature_records)
            if skipped_self_opponent_matches:
                logging.getLogger("tennis_pipeline").warning(
                    "Skipped self-opponent matches while building player_match_load_features | count=%s",
                    skipped_self_opponent_matches,
                )
        return summary

    def sync_live_results(
        self,
        *,
        results_url: str,
        match_type: str,
        surface: str,
        best_of: int,
        tourney_level: str,
        run_id: str,
    ) -> SyncSummary:
        from tennis_model.atp_results import fetch_atp_completed_results

        live_frame = fetch_atp_completed_results(
            results_url,
            match_type=match_type,
            surface=surface,
            best_of=best_of,
            tourney_level=tourney_level,
        )
        summary = SyncSummary(rows_read=int(len(live_frame)))
        now = _utc_now()
        upsert_sql = _upsert_sql("live_match_feed", LIVE_MATCH_COLUMNS, ["feed_key"])

        with self.connect() as connection:
            for raw_row in live_frame.to_dict("records"):
                row = {
                    key: None if pd.isna(value) else value
                    for key, value in raw_row.items()
                }
                feed_key = _live_feed_key(row)
                payload_json = _json_dumps(row)
                payload_hash = _payload_hash(payload_json)
                existing = connection.execute(
                    "SELECT payload_hash, first_seen_at FROM live_match_feed WHERE feed_key = ?",
                    (feed_key,),
                ).fetchone()
                if existing is None:
                    summary.inserted += 1
                    first_seen_at = now
                elif existing["payload_hash"] == payload_hash:
                    summary.unchanged += 1
                    connection.execute(
                        """
                        UPDATE live_match_feed
                        SET last_seen_at = ?, last_run_id = ?
                        WHERE feed_key = ?
                        """,
                        (now, run_id, feed_key),
                    )
                    continue
                else:
                    summary.updated += 1
                    first_seen_at = str(existing["first_seen_at"])

                record = (
                    feed_key,
                    _clean_identifier(row.get("match_id")),
                    payload_hash,
                    _clean_date(row.get("match_date")),
                    _clean_text(row.get("round_text")),
                    _clean_text(row.get("round_name")),
                    _clean_text(row.get("court_label")),
                    _clean_text(row.get("duration_text")),
                    _clean_text(row.get("player_1")),
                    _clean_text(row.get("player_2")),
                    _clean_text(row.get("winner")),
                    _clean_text(row.get("score_text")),
                    _clean_text(row.get("results_url")),
                    _clean_text(row.get("stats_url")),
                    _clean_text(row.get("h2h_url")),
                    _clean_text(row.get("surface")),
                    _clean_int(row.get("best_of")),
                    _clean_text(row.get("tourney_level")),
                    _clean_text(row.get("stats_fetch_error")),
                    *[_clean_int(row.get(column)) for column in LIVE_STAT_COLUMNS],
                    payload_json,
                    first_seen_at,
                    now,
                    run_id,
                )
                connection.execute(upsert_sql, record)
                connection.execute(
                    """
                    INSERT INTO live_match_snapshots (
                        run_id, feed_key, payload_hash, payload_json, captured_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (run_id, feed_key, payload_hash, payload_json, now),
                )
        return summary

    def sync_flashscore_match_urls(
        self,
        *,
        match_urls: Sequence[str],
        run_id: str,
    ) -> SyncSummary:
        from tennis_model.flashscore import fetch_match_snapshot

        unique_urls = [url for url in dict.fromkeys(match_urls) if _clean_text(url)]
        summary = SyncSummary(rows_read=len(unique_urls))
        if not unique_urls:
            return summary
        summary.synced_files = len(unique_urls)

        now = _utc_now()
        upsert_sql = _upsert_sql("flashscore_match_feed", FLASHSCORE_MATCH_COLUMNS, ["match_key"])

        with self.connect() as connection:
            for match_url in unique_urls:
                snapshot = fetch_match_snapshot(match_url)
                payload = snapshot.to_payload()
                match_key = _flashscore_match_key(payload.get("event_id"), payload.get("match_url"))
                payload_json = _json_dumps(payload)
                payload_hash = _payload_hash(payload_json)

                existing = connection.execute(
                    "SELECT payload_hash, first_seen_at FROM flashscore_match_feed WHERE match_key = ?",
                    (match_key,),
                ).fetchone()
                if existing is None:
                    summary.inserted += 1
                    first_seen_at = now
                elif existing["payload_hash"] == payload_hash:
                    summary.unchanged += 1
                    connection.execute(
                        """
                        UPDATE flashscore_match_feed
                        SET last_seen_at = ?, last_run_id = ?
                        WHERE match_key = ?
                        """,
                        (now, run_id, match_key),
                    )
                    continue
                else:
                    summary.updated += 1
                    first_seen_at = str(existing["first_seen_at"])

                record = (
                    match_key,
                    _clean_identifier(payload.get("event_id")),
                    payload_hash,
                    _clean_text(payload.get("match_url")),
                    _clean_text(payload.get("home_name")),
                    _clean_text(payload.get("away_name")),
                    _clean_identifier(payload.get("home_participant_id")),
                    _clean_identifier(payload.get("away_participant_id")),
                    _clean_text(payload.get("tournament_name")),
                    _clean_text(payload.get("category_name")),
                    _clean_text(payload.get("country_name")),
                    _clean_text(payload.get("point_score_home")),
                    _clean_text(payload.get("point_score_away")),
                    str(payload.get("current_game_feed_raw") or ""),
                    _json_dumps(payload.get("current_game_feed") or {}),
                    str(payload.get("common_feed_raw") or ""),
                    _json_dumps(payload.get("common_feed") or {}),
                    _clean_text(payload.get("match_history_feed_raw")),
                    _clean_text(payload.get("game_feed_raw")),
                    payload_json,
                    first_seen_at,
                    now,
                    run_id,
                )
                connection.execute(upsert_sql, record)
                connection.execute(
                    """
                    INSERT INTO flashscore_point_snapshots (
                        run_id,
                        match_key,
                        event_id,
                        payload_hash,
                        point_score_home,
                        point_score_away,
                        current_game_feed_raw,
                        current_game_feed_json,
                        common_feed_raw,
                        common_feed_json,
                        match_history_feed_raw,
                        game_feed_raw,
                        captured_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        run_id,
                        match_key,
                        _clean_identifier(payload.get("event_id")),
                        payload_hash,
                        _clean_text(payload.get("point_score_home")),
                        _clean_text(payload.get("point_score_away")),
                        str(payload.get("current_game_feed_raw") or ""),
                        _json_dumps(payload.get("current_game_feed") or {}),
                        str(payload.get("common_feed_raw") or ""),
                        _json_dumps(payload.get("common_feed") or {}),
                        _clean_text(payload.get("match_history_feed_raw")),
                        _clean_text(payload.get("game_feed_raw")),
                        now,
                    ),
                )
        return summary

    def sync_grand_slam_matches(
        self,
        *,
        slams: Sequence[str],
        run_id: str,
        include_match_pages: bool = False,
        rg_event_codes: Sequence[str] | None = None,
        season_years: Sequence[int] | None = None,
    ) -> SyncSummary:
        from tennis_model.grand_slam import (
            extract_point_events_from_snapshot,
            fetch_grand_slam_match_snapshots,
            snapshot_has_point_history,
        )

        snapshots = fetch_grand_slam_match_snapshots(
            slams=slams,
            include_match_pages=include_match_pages,
            rg_event_codes=rg_event_codes or ("SM",),
            season_years=season_years,
        )
        summary = SyncSummary(rows_read=len(snapshots), synced_files=len(snapshots))
        if not snapshots:
            return summary

        now = _utc_now()
        upsert_sql = _upsert_sql("grand_slam_match_feed", GRAND_SLAM_MATCH_COLUMNS, ["match_key"])

        with self.connect() as connection:
            for snapshot in snapshots:
                payload = snapshot.to_payload()
                match_key = _grand_slam_match_key(
                    payload.get("slam_code"),
                    payload.get("source_match_id"),
                    payload.get("match_url"),
                )
                payload_json = _json_dumps(payload)
                payload_hash = _payload_hash(payload_json)

                existing = connection.execute(
                    "SELECT payload_hash, first_seen_at FROM grand_slam_match_feed WHERE match_key = ?",
                    (match_key,),
                ).fetchone()
                if existing is None:
                    summary.inserted += 1
                    first_seen_at = now
                elif existing["payload_hash"] == payload_hash:
                    summary.unchanged += 1
                    connection.execute(
                        """
                        UPDATE grand_slam_match_feed
                        SET last_seen_at = ?, last_run_id = ?
                        WHERE match_key = ?
                        """,
                        (now, run_id, match_key),
                    )
                    continue
                else:
                    summary.updated += 1
                    first_seen_at = str(existing["first_seen_at"])

                record = (
                    match_key,
                    _clean_text(payload.get("slam_code")),
                    _clean_text(payload.get("provider_name")),
                    _clean_identifier(payload.get("source_match_id")),
                    _clean_int(payload.get("season_year")),
                    payload_hash,
                    _clean_text(payload.get("match_url")),
                    _clean_text(payload.get("event_name")),
                    _clean_text(payload.get("round_name")),
                    _clean_text(payload.get("court_name")),
                    _clean_text(payload.get("status_text")),
                    _clean_text(payload.get("status_code")),
                    _clean_text(payload.get("home_name")),
                    _clean_text(payload.get("away_name")),
                    _clean_text(payload.get("winner_side")),
                    _clean_text(payload.get("score_text")),
                    _clean_text(payload.get("point_score_home")),
                    _clean_text(payload.get("point_score_away")),
                    _clean_text(payload.get("discovery_source")),
                    _json_dumps(payload.get("schedule_payload") or {}),
                    _json_dumps(payload.get("detail_payload") or {}),
                    _json_dumps(payload.get("stats_payload") or {}),
                    _json_dumps(payload.get("history_payload") or {}),
                    _json_dumps(payload.get("keys_payload") or {}),
                    _json_dumps(payload.get("insights_payload") or {}),
                    _json_dumps(payload.get("page_payload") or {}),
                    payload_json,
                    first_seen_at,
                    now,
                    run_id,
                )
                connection.execute(upsert_sql, record)
                connection.execute(
                    """
                    INSERT INTO grand_slam_match_snapshots (
                        run_id,
                        match_key,
                        slam_code,
                        source_match_id,
                        payload_hash,
                        status_text,
                        score_text,
                        point_score_home,
                        point_score_away,
                        schedule_payload_json,
                        detail_payload_json,
                        stats_payload_json,
                        history_payload_json,
                        keys_payload_json,
                        insights_payload_json,
                        page_payload_json,
                        captured_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        run_id,
                        match_key,
                        _clean_text(payload.get("slam_code")),
                        _clean_identifier(payload.get("source_match_id")),
                        payload_hash,
                        _clean_text(payload.get("status_text")),
                        _clean_text(payload.get("score_text")),
                        _clean_text(payload.get("point_score_home")),
                        _clean_text(payload.get("point_score_away")),
                        _json_dumps(payload.get("schedule_payload") or {}),
                        _json_dumps(payload.get("detail_payload") or {}),
                        _json_dumps(payload.get("stats_payload") or {}),
                        _json_dumps(payload.get("history_payload") or {}),
                        _json_dumps(payload.get("keys_payload") or {}),
                        _json_dumps(payload.get("insights_payload") or {}),
                        _json_dumps(payload.get("page_payload") or {}),
                        now,
                    ),
                )
                if snapshot_has_point_history(snapshot):
                    point_events = extract_point_events_from_snapshot(snapshot, match_id=match_key)
                    connection.execute(
                        "DELETE FROM point_events WHERE match_id = ? AND source_system = ?",
                        (match_key, "grand_slam_official"),
                    )
                    if point_events:
                        point_records = [
                            (
                                row.match_id,
                                row.set_no,
                                row.game_no,
                                row.point_no,
                                _clean_text(row.server_id),
                                _clean_text(row.returner_id),
                                _clean_text(row.score_state),
                                int(bool(row.break_point_flag)),
                                int(bool(row.ace_flag)),
                                int(bool(row.winner_flag)),
                                int(bool(row.unforced_error_flag)),
                                _clean_int(row.rally_count),
                                _clean_int(row.serve_speed),
                                _clean_text(row.serve_direction),
                                _clean_text(row.return_depth),
                                _clean_text(row.point_winner_id),
                                "grand_slam_official",
                                now,
                            )
                            for row in point_events
                        ]
                        connection.executemany(
                            f"""
                            INSERT INTO point_events ({", ".join(POINT_EVENT_COLUMNS)})
                            VALUES ({", ".join("?" for _ in POINT_EVENT_COLUMNS)})
                            ON CONFLICT (match_id, set_no, game_no, point_no, source_system)
                            DO UPDATE SET
                                server_id = excluded.server_id,
                                returner_id = excluded.returner_id,
                                score_state = excluded.score_state,
                                break_point_flag = excluded.break_point_flag,
                                ace_flag = excluded.ace_flag,
                                winner_flag = excluded.winner_flag,
                                unforced_error_flag = excluded.unforced_error_flag,
                                rally_count = excluded.rally_count,
                                serve_speed = excluded.serve_speed,
                                serve_direction = excluded.serve_direction,
                                return_depth = excluded.return_depth,
                                point_winner_id = excluded.point_winner_id,
                                updated_at = excluded.updated_at
                            """,
                            point_records,
                        )
        return summary

    def table_counts(self) -> dict[str, int]:
        tables = [
            "players",
            "player_aliases",
            "player_rankings",
            "player_utr_ratings",
            "tournaments",
            "tournament_editions",
            "tournament_locations",
            "historical_matches",
            "matches",
            "player_match_stats",
            "player_match_load_features",
            "live_match_feed",
            "live_match_snapshots",
            "flashscore_match_feed",
            "flashscore_point_snapshots",
            "grand_slam_match_feed",
            "grand_slam_match_snapshots",
            "point_events",
        ]
        with self.connect() as connection:
            return {
                table: int(
                    connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
                )
                for table in tables
            }
