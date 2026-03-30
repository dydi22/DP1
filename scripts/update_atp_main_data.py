from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download newer ATP main-tour yearly files into the local data folder.")
    parser.add_argument("--data-dir", default="data/raw/tennis_atp", help="Directory where ATP CSV files live.")
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2025", "2026"],
        help="ATP yearly files to download from the TML database mirror.",
    )
    parser.add_argument(
        "--include-ongoing",
        action="store_true",
        help="Also download ongoing_tourneys.csv as atp_matches_2026_ongoing.csv.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    downloaded: list[Path] = []

    for year in args.years:
        url = f"{TML_BASE}/{year}.csv"
        destination = data_dir / f"atp_matches_{year}.csv"
        download_file(url, destination)
        downloaded.append(destination)

    if args.include_ongoing:
        destination = data_dir / "atp_matches_2026_ongoing.csv"
        download_file(f"{TML_BASE}/ongoing_tourneys.csv", destination)
        downloaded.append(destination)

    for path in downloaded:
        print(f"Downloaded {path}")


if __name__ == "__main__":
    main()
