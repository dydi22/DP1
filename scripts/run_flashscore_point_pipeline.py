from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_model.database import TennisDatabase
from tennis_model.flashscore import (
    DEFAULT_TENNIS_INDEX_URL,
    discover_current_atp_challenger_match_urls,
    discover_tennis_match_urls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll Flashscore tennis match pages and store live point-score snapshots "
            "into the DuckDB tennis database."
        )
    )
    parser.add_argument("--db-path", default="data/tennis_pipeline/tennis.duckdb")
    parser.add_argument(
        "--match-url",
        action="append",
        default=[],
        help="Flashscore tennis match URL. Repeat the flag to include multiple matches.",
    )
    parser.add_argument(
        "--match-urls-file",
        default=None,
        help="Optional text file with one Flashscore tennis match URL per line.",
    )
    parser.add_argument(
        "--discover-tennis-page",
        action="store_true",
        help="Discover match URLs from the Flashscore tennis landing page on each loop.",
    )
    parser.add_argument(
        "--discover-current-atp-challenger",
        action="store_true",
        help=(
            "Discover active ATP singles and Challenger men singles tournament pages from Flashscore, "
            "then collect their summary and fixtures match URLs."
        ),
    )
    parser.add_argument(
        "--discover-url",
        default=DEFAULT_TENNIS_INDEX_URL,
        help="Flashscore landing page used with --discover-tennis-page.",
    )
    parser.add_argument("--poll-seconds", type=int, default=None, help="Continuously poll on this cadence.")
    parser.add_argument("--max-loops", type=int, default=None, help="Optional loop cap when polling.")
    return parser.parse_args()


def _load_match_urls_file(path: Path) -> list[str]:
    urls: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            urls.append(cleaned)
    return urls


def resolve_match_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = list(args.match_url or [])
    if args.match_urls_file:
        urls.extend(_load_match_urls_file(Path(args.match_urls_file)))
    if args.discover_tennis_page:
        urls.extend(discover_tennis_match_urls(args.discover_url))
    if args.discover_current_atp_challenger:
        urls.extend(discover_current_atp_challenger_match_urls(args.discover_url))

    seen: set[str] = set()
    resolved: list[str] = []
    for url in urls:
        cleaned = url.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        resolved.append(cleaned)
    return resolved


def print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    database = TennisDatabase(args.db_path)
    database.initialize()

    iteration = 0
    while True:
        match_urls = resolve_match_urls(args)
        if not match_urls:
            raise SystemExit(
                "No Flashscore match URLs were provided. Use --match-url, --match-urls-file, "
                "--discover-tennis-page, or --discover-current-atp-challenger."
            )

        run_config = {
            "db_path": str(args.db_path),
            "match_urls": match_urls,
            "discover_tennis_page": args.discover_tennis_page,
            "discover_current_atp_challenger": args.discover_current_atp_challenger,
            "discover_url": args.discover_url,
            "poll_seconds": args.poll_seconds,
        }
        run_id = database.start_run("run_flashscore_point_pipeline", run_config)
        final_summary: dict[str, object]
        try:
            sync_summary = {
                "run_id": run_id,
                "match_urls": match_urls,
                "flashscore_point_stream": database.sync_flashscore_match_urls(
                    match_urls=match_urls,
                    run_id=run_id,
                ).to_dict(),
            }
            sync_summary["table_counts"] = database.table_counts()
            database.finish_run(run_id, "completed", sync_summary)
            final_summary = sync_summary
        except Exception as exc:
            final_summary = {
                "run_id": run_id,
                "status": "failed",
                "error": str(exc),
            }
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
