from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.database import TennisDatabase


TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
DEFAULT_UTR_CURRENT_PATTERN = "*.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and maintain a DuckDB tennis database from ATP historical files "
            "and optional live ATP results pages."
        )
    )
    parser.add_argument("--db-path", default="data/tennis_pipeline/tennis.duckdb")
    parser.add_argument("--data-dir", default="data/raw/tennis_atp")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to a .log file next to --db-path.",
    )
    parser.add_argument("--players-csv", default=None)
    parser.add_argument("--tournament-locations-csv", default=None, help="Optional CSV with tournament_name,city,country,timezone_name,utc_offset_hours,latitude,longitude.")
    parser.add_argument("--utr-history-csv", default=None, help="Historical UTR CSV with player_name,rating_date,utr_singles.")
    parser.add_argument("--utr-current-csv", default=None, help="Current weekly UTR CSV with player_name,utr_singles and optional rating_date.")
    parser.add_argument("--utr-current-dir", default=None, help="Directory containing weekly UTR snapshot CSVs. The pipeline imports the most recently modified file that matches --utr-current-pattern.")
    parser.add_argument("--utr-current-pattern", default=DEFAULT_UTR_CURRENT_PATTERN, help="Glob pattern used with --utr-current-dir to locate weekly UTR CSV snapshots.")
    parser.add_argument("--utr-alias-csv", default=None, help="Optional alias CSV with source_name,canonical_name for UTR name alignment.")
    parser.add_argument("--utr-rating-date", default=None, help="Default rating date for --utr-current-csv when the file does not include rating_date.")
    parser.add_argument("--refresh-utr-from-site", action="store_true", help="Fetch a fresh public UTR weekly snapshot from the UTR rankings/search endpoints before importing it.")
    parser.add_argument("--utr-site-output-dir", default="data/utr/weekly_exports", help="Directory where scraped weekly UTR site snapshots should be written.")
    parser.add_argument("--utr-site-tags", default="Pro", help="UTR public rankings tag to fetch, such as Pro.")
    parser.add_argument("--utr-site-gender", default="m", choices=["m", "f"], help="UTR public rankings gender code.")
    parser.add_argument("--utr-site-top-count", type=int, default=200, help="Number of players to pull from the public UTR rankings endpoint.")
    parser.add_argument("--utr-site-search-top", type=int, default=10, help="Number of search hits to inspect for each missing player.")
    parser.add_argument("--utr-site-workers", type=int, default=8, help="Worker count for missing-player UTR search lookups.")
    parser.add_argument("--utr-site-all-players", action="store_true", help="Search all local players instead of only players active in the recent window.")
    parser.add_argument("--utr-site-active-days", type=int, default=730, help="How many recent days of local match activity define the weekly UTR refresh set.")
    parser.add_argument("--utr-site-max-players", type=int, default=None, help="Optional cap for local players included in a UTR site refresh.")
    parser.add_argument("--utr-site-no-search-missing", dest="utr_site_search_missing", action="store_false", help="Disable per-player public search lookups for local players not found in the top rankings sweep.")
    parser.set_defaults(utr_site_search_missing=True)
    parser.add_argument("--years", nargs="*", default=None, help="Main-tour ATP years to sync. Defaults to all yearly files found.")
    parser.add_argument("--include-ongoing", action="store_true", help="Include ATP ongoing match files when present.")
    parser.add_argument(
        "--ranking-files",
        nargs="*",
        default=None,
        help="Optional ranking filenames relative to --data-dir, such as atp_rankings_current.csv.",
    )
    parser.add_argument("--skip-players", action="store_true")
    parser.add_argument("--skip-player-aliases", action="store_true")
    parser.add_argument("--skip-rankings", action="store_true")
    parser.add_argument("--skip-utr", action="store_true")
    parser.add_argument("--skip-tournament-locations", action="store_true")
    parser.add_argument("--skip-historical", action="store_true")
    parser.add_argument("--skip-tournament-dimensions", action="store_true")
    parser.add_argument("--skip-normalized-matches", action="store_true")
    parser.add_argument("--skip-player-match-stats", action="store_true")
    parser.add_argument("--skip-load-features", action="store_true")
    parser.add_argument("--load-weight-minutes-last-7d", type=float, default=0.35)
    parser.add_argument("--load-weight-matches-last-7d", type=float, default=0.35)
    parser.add_argument("--load-weight-travel-timezones-last-7d", type=float, default=0.20)
    parser.add_argument("--load-weight-tiebreaks-last-7d", type=float, default=0.10)
    parser.add_argument("--force", action="store_true", help="Re-sync even when source file timestamps have not changed.")
    parser.add_argument("--refresh-main-data", action="store_true", help="Download fresh ATP main-tour yearly files before syncing.")
    parser.add_argument(
        "--refresh-years",
        nargs="+",
        default=["2025", "2026"],
        help="Years to download when --refresh-main-data is enabled.",
    )
    parser.add_argument(
        "--results-url",
        default=None,
        help="Optional ATP results page to poll for completed matches and official stats.",
    )
    parser.add_argument("--results-match-type", default="singles")
    parser.add_argument("--surface", default=None, help="Required with --results-url.")
    parser.add_argument("--best-of", type=int, default=None, help="Required with --results-url.")
    parser.add_argument("--tourney-level", default=None, help="Required with --results-url.")
    parser.add_argument("--poll-seconds", type=int, default=None, help="Continuously poll the live results page on this cadence.")
    parser.add_argument("--max-loops", type=int, default=None, help="Optional loop cap when --poll-seconds is set.")
    return parser.parse_args()


def download_main_tour_files(data_dir: Path, years: list[str], include_ongoing: bool) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for year in years:
        destination = data_dir / f"atp_matches_{year}.csv"
        urlretrieve(f"{TML_BASE}/{year}.csv", destination)
        downloaded.append(destination)
    if include_ongoing:
        destination = data_dir / "atp_matches_2026_ongoing.csv"
        urlretrieve(f"{TML_BASE}/ongoing_tourneys.csv", destination)
        downloaded.append(destination)
    return downloaded


def print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def configure_logging(log_file: str | Path) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tennis_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_step_start(logger: logging.Logger, step_name: str) -> float:
    logger.info("Starting step: %s", step_name)
    return time.perf_counter()


def log_step_end(logger: logging.Logger, step_name: str, started_at: float, summary: dict[str, object]) -> None:
    elapsed = time.perf_counter() - started_at
    logger.info(
        "Finished step: %s in %.2fs | summary=%s",
        step_name,
        elapsed,
        json.dumps(summary, sort_keys=True),
    )


def resolve_latest_utr_current_csv(directory: str | Path, pattern: str) -> Path:
    base_dir = Path(directory)
    if not base_dir.exists():
        raise FileNotFoundError(f"UTR current directory not found: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"UTR current path is not a directory: {base_dir}")

    candidates = [path for path in base_dir.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No UTR current CSV files matched pattern {pattern!r} in {base_dir}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def main() -> None:
    args = parse_args()
    if args.results_url and (args.surface is None or args.best_of is None or args.tourney_level is None):
        raise SystemExit("--results-url requires --surface, --best-of, and --tourney-level.")
    utr_sources = [bool(args.utr_current_csv), bool(args.utr_current_dir), bool(args.refresh_utr_from_site)]
    if sum(utr_sources) > 1:
        raise SystemExit("Use only one UTR input source: --utr-current-csv, --utr-current-dir, or --refresh-utr-from-site.")

    data_dir = Path(args.data_dir)
    players_csv = Path(args.players_csv) if args.players_csv else data_dir / "atp_players.csv"
    db_path = Path(args.db_path)
    log_file = args.log_file or db_path.with_suffix(".log")
    logger = configure_logging(log_file)
    logger.info("Pipeline starting | db_path=%s data_dir=%s", db_path, data_dir)

    if args.refresh_main_data:
        downloaded = download_main_tour_files(data_dir, args.refresh_years, args.include_ongoing)
        print_summary({"downloaded_files": [str(path) for path in downloaded]})

    database = TennisDatabase(db_path)
    logger.info("Initializing database schema and metadata")
    try:
        database.initialize()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted during initialization")
        raise SystemExit(130)
    except Exception:
        logger.exception("Pipeline failed during initialization")
        raise

    iteration = 0
    while True:
        resolved_utr_current_csv = args.utr_current_csv
        if args.utr_current_dir:
            resolved_utr_current_csv = str(
                resolve_latest_utr_current_csv(args.utr_current_dir, args.utr_current_pattern)
            )
        run_config = {
            "db_path": str(args.db_path),
            "data_dir": str(data_dir),
            "utr_history_csv": args.utr_history_csv,
            "utr_current_csv": resolved_utr_current_csv,
            "utr_current_dir": args.utr_current_dir,
            "utr_current_pattern": args.utr_current_pattern,
            "refresh_utr_from_site": args.refresh_utr_from_site,
            "tournament_locations_csv": args.tournament_locations_csv,
            "years": args.years,
            "include_ongoing": args.include_ongoing,
            "results_url": args.results_url,
            "poll_seconds": args.poll_seconds,
            "load_score_weights": {
                "minutes_last_7d": args.load_weight_minutes_last_7d,
                "matches_last_7d": args.load_weight_matches_last_7d,
                "travel_timezones_last_7d": args.load_weight_travel_timezones_last_7d,
                "tiebreaks_last_7d": args.load_weight_tiebreaks_last_7d,
            },
        }
        run_id = database.start_run("run_tennis_database_pipeline", run_config)
        final_summary: dict[str, object]
        sync_summary: dict[str, object] = {"run_id": run_id}
        current_step = "initializing"
        try:
            if not args.skip_players:
                current_step = "players"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["players"] = database.sync_players_file(players_csv, force=args.force).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["players"])
            if not args.skip_rankings:
                current_step = "rankings"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["rankings"] = database.sync_ranking_files(
                    data_dir,
                    ranking_files=args.ranking_files,
                    force=args.force,
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["rankings"])
            if not args.skip_utr and args.utr_history_csv:
                current_step = "utr_history"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["utr_history"] = database.sync_utr_file(
                    args.utr_history_csv,
                    alias_csv=args.utr_alias_csv,
                    force=args.force,
                    source_type="utr_history_csv",
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["utr_history"])
            if not args.skip_utr and args.refresh_utr_from_site:
                from tennis_model.utr_site import scrape_public_utr_snapshot

                current_step = "utr_site_scrape"
                step_started_at = log_step_start(logger, current_step)
                active_since = None
                if not args.utr_site_all_players:
                    active_since = (
                        datetime.now().date() - timedelta(days=max(args.utr_site_active_days, 0))
                    ).isoformat()
                tracked_players = database.players_for_utr_site_refresh(
                    active_since=active_since,
                    limit=args.utr_site_max_players,
                )
                if not tracked_players and active_since is not None:
                    tracked_players = database.players_for_utr_site_refresh(limit=args.utr_site_max_players)
                scraped_csv, scrape_summary = scrape_public_utr_snapshot(
                    output_dir=args.utr_site_output_dir,
                    rating_date=args.utr_rating_date or datetime.now().date().isoformat(),
                    tracked_players=tracked_players,
                    gender=args.utr_site_gender,
                    tags=args.utr_site_tags,
                    top_count=args.utr_site_top_count,
                    search_missing=args.utr_site_search_missing,
                    search_top=args.utr_site_search_top,
                    workers=args.utr_site_workers,
                )
                resolved_utr_current_csv = str(scraped_csv)
                sync_summary["utr_site_scrape"] = scrape_summary.to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["utr_site_scrape"])
            if not args.skip_utr and args.utr_current_csv:
                current_step = "utr_current"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["utr_current"] = database.sync_utr_file(
                    args.utr_current_csv,
                    alias_csv=args.utr_alias_csv,
                    default_rating_date=args.utr_rating_date or datetime.now().date().isoformat(),
                    force=args.force,
                    source_type="utr_current_csv",
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["utr_current"])
            if not args.skip_utr and args.utr_current_dir:
                current_step = "utr_current"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["utr_current"] = database.sync_utr_file(
                    resolved_utr_current_csv,
                    alias_csv=args.utr_alias_csv,
                    default_rating_date=args.utr_rating_date or datetime.now().date().isoformat(),
                    force=args.force,
                    source_type="utr_current_csv",
                ).to_dict()
                sync_summary["utr_current_source_file"] = resolved_utr_current_csv
                log_step_end(logger, current_step, step_started_at, sync_summary["utr_current"])
            if not args.skip_utr and args.refresh_utr_from_site:
                current_step = "utr_current"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["utr_current"] = database.sync_utr_file(
                    resolved_utr_current_csv,
                    alias_csv=args.utr_alias_csv,
                    default_rating_date=args.utr_rating_date or datetime.now().date().isoformat(),
                    force=args.force,
                    source_type="utr_current_csv",
                ).to_dict()
                sync_summary["utr_current_source_file"] = resolved_utr_current_csv
                log_step_end(logger, current_step, step_started_at, sync_summary["utr_current"])
            if not args.skip_player_aliases:
                current_step = "player_aliases"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["player_aliases"] = database.sync_player_aliases(
                    alias_csv=args.utr_alias_csv,
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["player_aliases"])
            if not args.skip_tournament_locations and args.tournament_locations_csv:
                current_step = "tournament_locations"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["tournament_locations"] = database.sync_tournament_locations_file(
                    args.tournament_locations_csv,
                    force=args.force,
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["tournament_locations"])
            if not args.skip_historical:
                current_step = "historical_matches"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["historical_matches"] = database.sync_historical_matches(
                    data_dir,
                    years=args.years,
                    include_ongoing=args.include_ongoing,
                    force=args.force,
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["historical_matches"])
            if not args.skip_tournament_dimensions:
                current_step = "tournament_dimensions"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["tournament_dimensions"] = database.sync_tournament_dimensions().to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["tournament_dimensions"])
            if not args.skip_normalized_matches:
                current_step = "matches"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["matches"] = database.sync_matches_table().to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["matches"])
            if not args.skip_player_match_stats:
                current_step = "player_match_stats"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["player_match_stats"] = database.sync_player_match_stats().to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["player_match_stats"])
            if not args.skip_load_features:
                current_step = "player_match_load_features"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["player_match_load_features"] = database.sync_player_match_load_features(
                    load_weight_minutes_last_7d=args.load_weight_minutes_last_7d,
                    load_weight_matches_last_7d=args.load_weight_matches_last_7d,
                    load_weight_travel_timezones_last_7d=args.load_weight_travel_timezones_last_7d,
                    load_weight_tiebreaks_last_7d=args.load_weight_tiebreaks_last_7d,
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["player_match_load_features"])
            if args.results_url:
                current_step = "live_results"
                step_started_at = log_step_start(logger, current_step)
                sync_summary["live_results"] = database.sync_live_results(
                    results_url=args.results_url,
                    match_type=args.results_match_type,
                    surface=args.surface,
                    best_of=args.best_of,
                    tourney_level=args.tourney_level,
                    run_id=run_id,
                ).to_dict()
                log_step_end(logger, current_step, step_started_at, sync_summary["live_results"])

            current_step = "table_counts"
            step_started_at = log_step_start(logger, current_step)
            sync_summary["table_counts"] = database.table_counts()
            log_step_end(logger, current_step, step_started_at, sync_summary["table_counts"])
            database.finish_run(run_id, "completed", sync_summary)
            final_summary = sync_summary
            logger.info("Pipeline completed | run_id=%s", run_id)
        except KeyboardInterrupt:
            final_summary = {
                "run_id": run_id,
                "status": "interrupted",
                "current_step": current_step,
                "partial_summary": sync_summary,
            }
            logger.warning("Pipeline interrupted during step: %s", current_step)
            database.finish_run(run_id, "interrupted", final_summary)
            print_summary(final_summary)
            raise SystemExit(130)
        except Exception as exc:
            final_summary = {
                "run_id": run_id,
                "status": "failed",
                "current_step": current_step,
                "error": str(exc),
                "partial_summary": sync_summary,
            }
            logger.exception("Pipeline failed during step: %s", current_step)
            database.finish_run(run_id, "failed", final_summary)
            raise

        print_summary(final_summary)
        iteration += 1
        if not args.poll_seconds:
            break
        if args.max_loops is not None and iteration >= args.max_loops:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
