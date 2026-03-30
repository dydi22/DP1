from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.database import TennisDatabase

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - environment-dependent
    NotOpenSSLWarning = None


DEFAULT_SLAMS = ["wimbledon", "us_open", "australian_open", "roland_garros"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest official Grand Slam match feeds into the DuckDB tennis database. "
            "Wimbledon and the US Open use their official structured scoring feeds, "
            "while the Australian Open and Roland-Garros use their official results and match pages."
        )
    )
    parser.add_argument("--db-path", default="data/tennis_pipeline/tennis.duckdb")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to a .log file next to --db-path.",
    )
    parser.add_argument(
        "--slam",
        action="append",
        choices=DEFAULT_SLAMS,
        default=[],
        help="Grand Slam source to ingest. Repeat the flag to limit the run to specific slams.",
    )
    parser.add_argument(
        "--include-match-pages",
        action="store_true",
        help="Also fetch official match page metadata where the slam site exposes a match page.",
    )
    parser.add_argument(
        "--rg-event-code",
        action="append",
        default=[],
        help=(
            "Roland-Garros results event code to ingest. Repeat the flag for more than one code. "
            "Defaults to SM."
        ),
    )
    parser.add_argument(
        "--season-year",
        action="append",
        type=int,
        default=[],
        help=(
            "Historical Grand Slam season year to ingest. Repeat the flag for multiple years. "
            "This is currently strongest for Wimbledon and the US Open official point-history feeds."
        ),
    )
    return parser.parse_args()


def print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def configure_logging(log_file: str | Path) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("grand_slam_pipeline")
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


def main() -> None:
    if NotOpenSSLWarning is not None:
        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    args = parse_args()
    slams = args.slam or DEFAULT_SLAMS
    rg_event_codes = args.rg_event_code or ["SM"]
    db_path = Path(args.db_path)
    log_file = args.log_file or db_path.with_name(f"{db_path.stem}_grand_slam.log")
    logger = configure_logging(log_file)
    logger.info("Grand Slam pipeline starting | db_path=%s slams=%s", db_path, ",".join(slams))

    database = TennisDatabase(db_path)
    logger.info("Initializing database schema and metadata")
    try:
        database.initialize()
    except KeyboardInterrupt:
        logger.warning("Grand Slam pipeline interrupted during initialization")
        raise SystemExit(130)
    except Exception:
        logger.exception("Grand Slam pipeline failed during initialization")
        raise

    run_config = {
        "db_path": str(args.db_path),
        "slams": slams,
        "include_match_pages": args.include_match_pages,
        "rg_event_codes": rg_event_codes,
        "season_years": args.season_year,
    }
    run_id = database.start_run("run_grand_slam_pipeline", run_config)
    final_summary: dict[str, object]
    sync_summary: dict[str, object] = {"run_id": run_id, "slams": slams}
    current_step = "initializing"
    try:
        current_step = "grand_slam_official"
        step_started_at = log_step_start(logger, current_step)
        sync_summary["grand_slam_official"] = database.sync_grand_slam_matches(
                slams=slams,
                run_id=run_id,
                include_match_pages=args.include_match_pages,
                rg_event_codes=rg_event_codes,
                season_years=args.season_year,
            ).to_dict()
        log_step_end(logger, current_step, step_started_at, sync_summary["grand_slam_official"])
        current_step = "table_counts"
        step_started_at = log_step_start(logger, current_step)
        sync_summary["table_counts"] = database.table_counts()
        log_step_end(logger, current_step, step_started_at, sync_summary["table_counts"])
        database.finish_run(run_id, "completed", sync_summary)
        final_summary = sync_summary
        logger.info("Grand Slam pipeline completed | run_id=%s", run_id)
    except KeyboardInterrupt:
        final_summary = {
            "run_id": run_id,
            "status": "interrupted",
            "current_step": current_step,
            "partial_summary": sync_summary,
        }
        logger.warning("Grand Slam pipeline interrupted during step: %s", current_step)
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
        logger.exception("Grand Slam pipeline failed during step: %s", current_step)
        database.finish_run(run_id, "failed", final_summary)
        raise

    print_summary(final_summary)


if __name__ == "__main__":
    main()
