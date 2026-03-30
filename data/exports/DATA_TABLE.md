### Data Table

| Table | Description | Link |
|---|---|---|
| `players` | Master player table containing player identity and latest profile fields | [players.csv](./data/exports/players.csv) |
| `player_rankings` | Historical ATP ranking records by player and ranking date | [player_rankings.csv](./data/exports/player_rankings.csv) |
| `player_utr_ratings` | Historical UTR ratings by player and date | [player_utr_ratings.csv](./data/exports/player_utr_ratings.csv) |
| `player_aliases` | Cross-source player identity mapping table | [player_aliases.csv](./data/exports/player_aliases.csv) |
| `tournaments` | Canonical tournament dimension table | [tournaments.csv](./data/exports/tournaments.csv) |
| `tournament_editions` | Tournament season/year editions with location and surface context | [tournament_editions.csv](./data/exports/tournament_editions.csv) |
| `matches` | Match fact table containing match-level context and identifiers | [matches.csv](./data/exports/matches.csv) |
| `player_match_stats` | Per-player match-level statistics derived from historical matches | [player_match_stats.csv](./data/exports/player_match_stats.csv) |
| `player_match_load_features` | Per-player rolling workload and travel feature table | [player_match_load_features.csv](./data/exports/player_match_load_features.csv) |
| `point_events` | Point-by-point event table for supported match sources | [point_events.csv](./data/exports/point_events.csv) |
