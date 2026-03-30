# Tennis Match Predictor

This project trains leakage-safe ATP singles match models from the Jeff Sackmann `tennis_atp` dataset, benchmarks several model types, predicts `P(player_1 wins)`, and simulates knockout tournament brackets.

## What it does

- Trains and benchmarks multiple classifiers for `player_1 wins`
- Uses only information that would have been known before each match
- Handles all ATP singles matches across all surfaces
- Includes `best_of` so Grand Slams and best-of-3 events are treated differently
- Predicts single matches and full tournament brackets
- Supports optional historical UTR inputs when you have a legal data source
- Can also fetch a weekly public UTR snapshot from the UTR rankings/search site
- Builds richer tennis ratings including overall Elo, surface Elo, recent Elo, best-of-5 Elo, and inactivity-aware decay
- Uses uncertainty-aware Elo movement so newer or long-inactive players can move faster
- Adds rolling serve/return form features from prior matches only
- Benchmarks a direct Elo probability baseline and a soft-voting ensemble
- Saves a calibrated final model for better probabilities

## Recommended approach

The project now benchmarks:

- decision tree
- logistic regression
- random forest
- histogram gradient boosting

Model selection is done on a validation season, and the chosen model is then trained on the full dataset for production-style predictions.

## Data source

Clone the ATP repo into `data/raw/tennis_atp`:

```bash
mkdir -p data/raw
git clone https://github.com/JeffSackmann/tennis_atp.git data/raw/tennis_atp
```

Expected files include yearly ATP match CSVs such as `atp_matches_2024.csv`.

You can also extend the local history with newer ATP yearly files such as:

- `atp_matches_2025.csv`
- `atp_matches_2026.csv`
- `atp_matches_2026_ongoing.csv`

If those files are present in the same folder, the training and live-state scripts will pick them up automatically.

You can download those ATP main-tour files with:

```bash
python scripts/update_atp_main_data.py --include-ongoing
```

## Database pipeline

The repo now includes a DuckDB ingestion pipeline so match data can flow into a database instead of living only in CSV artifacts.

Build or refresh a local tennis database from the ATP files already in `data/raw/tennis_atp`:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --years 2024 2025 2026 \
  --include-ongoing
```

The pipeline now writes step-by-step logs to a `.log` file next to the database by default. You can override that with `--log-file`.

For example, if your database path is `data/tennis_pipeline/tennis.duckdb`, the default log file is `data/tennis_pipeline/tennis.log`.

During a long historical build, you can watch progress with:

```bash
tail -f data/tennis_pipeline/tennis.log
```

If you stop the run with `Ctrl+C`, the pipeline records the run as `interrupted` and stores the current step plus the partial sync summary in `ingestion_runs`.

That syncs:

- `players` from `atp_players.csv`
- `player_rankings` from `atp_rankings*.csv`
- `player_aliases`, a canonical alias table for cross-source player identity resolution
- `player_utr_ratings` from optional UTR history or weekly UTR snapshot files
- `tournaments`, a canonical tournament dimension keyed across recurring events
- `tournament_editions`, one row per tournament season/year edition with location and surface context
- `tournament_locations` from an optional tournament geography CSV
- `historical_matches` from the ATP main-tour yearly files
- `matches`, a normalized match fact table with `match_id`, `tournament_id`, `edition_id`, `start_time_utc`, `end_time_utc`, `duration_minutes`, `surface`, `best_of`, `tournament`, and `round`
- `player_match_stats`, a canonical per-player match stats table derived from the ATP winner/loser rows
- `player_match_load_features`, a per-player rolling load table built from the normalized match history
- `player_profile_view`, a one-row-per-player convenience view for latest ranking, UTR, match, and load context
- `player_timeline_view`, a player event stream view spanning ATP rankings, UTR ratings, and player-match events
- `unified_match_stream`, a database view that combines historical and live match feeds

Date formatting is normalized across the main pipeline tables:

- dates use ISO `YYYY-MM-DD`
- UTC timestamps use ISO `YYYY-MM-DDTHH:MM:SSZ`

`player_match_load_features` currently includes:

- `player_minutes_last_1_match`
- `player_minutes_last_3_matches`
- `player_minutes_last_7d`
- `player_minutes_last_14d`
- `player_avg_match_minutes_last_30d`
- `player_long_match_flag_last_match`
- `back_to_back_long_match_count`
- `player_matches_last_7d`
- `opponent_minutes_last_7d`
- `opponent_matches_last_7d`
- `minutes_diff_last_7d`
- `travel_timezones_last_7d`
- `tiebreaks_last_7d`
- `load_score`

`player_match_stats` currently includes:

- `match_id`
- `player_id`
- `opponent_id`
- `is_winner`
- `side`
- `ace`
- `df`
- `svpt`
- `first_in`
- `first_won`
- `second_won`
- `service_games`
- `bp_saved`
- `bp_faced`
- `minutes`

For ATP historical rows, `side` is `winner` or `loser`, because the source files are winner/loser oriented rather than true `player1` / `player2` order.

`player_aliases` currently includes:

- `alias_lookup`
- `alias_name`
- `canonical_player_id`
- `canonical_player_name`
- `source_system`
- `confidence_score`

The pipeline auto-builds self-aliases from `players` and also stores explicit UTR alias mappings when `--utr-alias-csv` is supplied or when `player_utr_ratings` contains a different source name and canonical name.

`player_utr_ratings` now also stores `canonical_player_id` so historical UTR rows can join directly to `players` instead of relying only on text lookups.

`tournaments` currently includes:

- `tournament_id`
- `canonical_tournament_name`
- `tour_level`
- `default_surface`
- `default_indoor`
- `country`
- `city`

`tournament_editions` currently includes:

- `edition_id`
- `tournament_id`
- `season_year`
- `source_tourney_id`
- `tournament_name`
- `start_date`
- `end_date`
- `city`
- `country`
- `timezone_name`
- `latitude`
- `longitude`
- `surface`
- `indoor`
- `draw_size`

Right now `start_date` comes from the ATP historical tournament date, while `end_date` stays null until a richer schedule source is added.

The travel layer in `player_match_load_features` now also stores raw pre-match travel context:

- `prev_tournament_city`
- `current_tournament_city`
- `prev_match_end_utc`
- `current_match_start_utc`
- `timezones_crossed_signed`
- `travel_direction`
- `great_circle_km`
- `hours_between_matches`
- `days_rest`
- `back_to_back_week_flag`

And engineered travel features:

- `tz_shift_signed`
- `tz_shift_abs`
- `eastward_shift_flag`
- `short_recovery_after_travel`
- `travel_fatigue_score`
- `load_plus_travel_score`
- `eastward_shift_minutes_last_7d`
- `eastward_shift_days_rest`
- `tz_shift_abs_previous_match_duration`
- `back_to_back_week_tz_shift_abs`

The current `load_score` formula is tunable from the CLI and defaults to:

```text
0.35 * player_minutes_last_7d
+ 0.35 * player_matches_last_7d
+ 0.20 * travel_timezones_last_7d
+ 0.10 * tiebreaks_last_7d
```

Right now `travel_timezones_last_7d` is a placeholder `0.0` until we add tournament geo and travel segments, so you can already run simulations on the weights and later fold real travel into the same table.

To turn on real timezone and distance travel features, pass a tournament location CSV such as [data/tournament_locations_template.csv](/Users/dylandietrich/Documents/New project/data/tournament_locations_template.csv):

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --tournament-locations-csv data/tournament_locations.csv
```

If you have UTR data, you can also attach it directly to the player database records. The pipeline stores the full UTR history in `player_utr_ratings` and keeps each player’s latest UTR in `players.latest_utr_singles` and `players.latest_utr_rating_date`.

The UTR tables also support optional weekly public-site fields when they are available:

- `utr_rank`
- `three_month_rating`
- `nationality`
- `provider_player_id`

Full historical UTR import:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --utr-history-csv data/utr/utr_history.csv \
  --utr-alias-csv data/utr/utr_aliases.csv \
  --skip-historical
```

Weekly UTR refresh from a current export:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --skip-rankings \
  --skip-historical \
  --utr-current-csv data/utr/utr_current.csv \
  --utr-alias-csv data/utr/utr_aliases.csv
```

If you want the pipeline to automatically grab the newest weekly UTR export from a folder, drop each new CSV into [data/utr/weekly_exports](/Users/dylandietrich/Documents/New project/data/utr/weekly_exports) and run:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --skip-rankings \
  --skip-historical \
  --utr-current-dir data/utr/weekly_exports \
  --utr-current-pattern "*.csv" \
  --utr-alias-csv data/utr/utr_aliases.csv
```

In this mode the runner picks the most recently modified matching CSV and reports it back as `utr_current_source_file` in the run summary.

You can also let the pipeline fetch a fresh weekly public UTR snapshot itself from the UTR rankings/search endpoints, write a dated CSV into [data/utr/weekly_exports](/Users/dylandietrich/Documents/New project/data/utr/weekly_exports), and then import that snapshot into the database:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --refresh-utr-from-site \
  --utr-site-output-dir data/utr/weekly_exports
```

By default this public-site refresh:

- fetches the public `Pro` rankings list for the selected gender
- uses recent local players as the weekly refresh set
- falls back to public player search for local players not found in the top rankings sweep
- writes the chosen snapshot path back as `utr_current_source_file`

For the ATP-focused database, the default public-site settings are aimed at men’s pro tennis. You can change them with flags like `--utr-site-gender`, `--utr-site-tags`, `--utr-site-active-days`, `--utr-site-max-players`, and `--utr-site-no-search-missing`.

`--utr-current-csv` accepts:

- `player_name,utr_singles` and uses the current date as the snapshot date
- or `player_name,rating_date,utr_singles` if you want to provide the snapshot date explicitly

To poll an ATP results page and keep pushing newly completed matches plus official match stats into the same database:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --skip-rankings \
  --skip-historical \
  --results-url "https://www.atptour.com/en/scores/current/miami/403/results" \
  --surface Hard \
  --best-of 3 \
  --tourney-level M \
  --poll-seconds 300
```

Live polling writes the latest state into `live_match_feed` and appends unique payload versions into `live_match_snapshots`, which gives you an append-only stream history for tournament updates.

You can tune the load-score weights during the build:

```bash
python scripts/run_tennis_database_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --data-dir data/raw/tennis_atp \
  --load-weight-minutes-last-7d 0.40 \
  --load-weight-matches-last-7d 0.25 \
  --load-weight-travel-timezones-last-7d 0.20 \
  --load-weight-tiebreaks-last-7d 0.15
```

There is also a Flashscore point-stream ingestor for live tennis match pages. This first version is designed for "from now onward" capture: it polls Flashscore match URLs, stores the raw internal feed payloads, and appends a new snapshot whenever the live point score changes.

Poll specific Flashscore match URLs:

```bash
python scripts/run_flashscore_point_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --match-url "https://www.flashscore.com/match/tennis/arthur-fils-IoIhUqIN/lehecka-jiri-6PlgfXKR/" \
  --match-url "https://www.flashscore.com/match/tennis/sinner-jannik-6HdC3z4H/zverev-alexander-dGbUhw9m/" \
  --poll-seconds 15
```

Or let the script discover currently linked tennis matches from the Flashscore tennis landing page each loop:

```bash
python scripts/run_flashscore_point_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --discover-tennis-page \
  --poll-seconds 15
```

For a broader ATP/challenger auto-tracking mode, let the script discover the current ATP singles and Challenger men singles tournaments from Flashscore first, then collect match URLs from each tournament's summary and fixtures pages:

```bash
python scripts/run_flashscore_point_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --discover-current-atp-challenger \
  --poll-seconds 15
```

This writes:

- `flashscore_match_feed` for the latest per-match state
- `flashscore_point_snapshots` for the append-only point-score stream history

The stored `point_score_home` and `point_score_away` columns are derived from Flashscore's live current-game feed. The pipeline also stores the raw `current_game_feed_raw`, `common_feed_raw`, `match_history_feed_raw`, and parsed JSON blobs so we can safely expand the parser later without losing data collected today.

There is also a built-in Grand Slam scraper for official slam feeds and match pages. It writes the latest official state for each match into `grand_slam_match_feed` and appends every unique payload version into `grand_slam_match_snapshots`.

When the official slam history feed exposes point-by-point rows, the pipeline also normalizes them into `point_events` with one row per point. The current point schema is:

- `match_id`
- `set_no`
- `game_no`
- `point_no`
- `server_id`
- `returner_id`
- `score_state`
- `break_point_flag`
- `ace_flag`
- `winner_flag`
- `unforced_error_flag`
- `rally_count`
- `serve_speed`
- `serve_direction`
- `return_depth`
- `point_winner_id`

Right now `point_events` is populated from the official Grand Slam history feeds where that level of detail is available, especially Wimbledon and the US Open. If a source does not expose a point history row for a field, the pipeline leaves it null instead of inventing data.

Run all four slams in one pass:

```bash
python scripts/run_grand_slam_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb
```

Include official match page metadata when the slam site exposes a match page:

```bash
python scripts/run_grand_slam_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --include-match-pages
```

Limit the run to a specific slam:

```bash
python scripts/run_grand_slam_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --slam wimbledon \
  --slam us_open
```

Backfill historical official Grand Slam point-by-point for specific seasons:

```bash
python scripts/run_grand_slam_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --slam wimbledon \
  --slam us_open \
  --season-year 2024 \
  --season-year 2025
```

For Roland-Garros, you can override the official results event codes to crawl:

```bash
python scripts/run_grand_slam_pipeline.py \
  --db-path data/tennis_pipeline/tennis.duckdb \
  --slam roland_garros \
  --rg-event-code SM
```

Current source coverage:

- `wimbledon`: official structured scores feeds, completed match lists, match-history feed, and optional match insights where exposed
- `us_open`: official structured scores feeds, completed match lists, match-history feed, and optional match insights where exposed
- `australian_open`: official results feeds discovered from the AO results page, plus optional match page metadata
- `roland_garros`: official results pages and official match page metadata based on discovered match URLs

Historical year backfill is currently strongest for `wimbledon` and `us_open`, because those official sites expose year-addressable completed-match and point-history feeds. The Australian Open and Roland-Garros are still useful crawl sources, but their currently wired endpoints are not yet as complete for historical point-by-point.

The Grand Slam ingestor stores raw JSON payloads for schedule, detail, stats, history, keys, insights, and page metadata where available so we can expand parsing later without losing the source snapshots.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib
```

For the Miami dashboard:

```bash
pip install streamlit plotly
```

For automated ATP draw imports:

```bash
pip install requests beautifulsoup4 lxml
```

## Train the model

```bash
python scripts/train_model.py --data-dir data/raw/tennis_atp
```

By default, the script uses:

- latest season as test data
- previous season as validation data
- all earlier seasons as training data

The benchmark now includes:

- `elo_probability`
- `rating_logistic_baseline`
- `logistic_regression`
- `decision_tree`
- `random_forest`
- `hist_gradient_boosting`
- `soft_voting_ensemble`

The current Elo update logic is now more chess-like:

- lower-experience players move faster
- long-inactive players get both rating decay and a larger post-layoff K-factor
- best-of-5 matches get a separate context adjustment

You can now tune those Elo controls directly:

- base `K`
- overall inactivity half-life
- recent Elo half-life
- best-of-5 Elo half-life
- inactivity max K boost
- recent Elo K multiplier

You can override the split:

```bash
python scripts/train_model.py \
  --data-dir data/raw/tennis_atp \
  --validation-start-year 2022 \
  --test-start-year 2023
```

If the latest season is only partially available, a more stable setup is often:

```bash
python scripts/train_model.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_plus_stats_2026_model \
  --validation-start-year 2024 \
  --test-start-year 2025
```

If you have historical UTR data:

```bash
python scripts/train_model.py \
  --data-dir data/raw/tennis_atp \
  --utr-history-csv data/utr/utr_history.csv \
  --utr-alias-csv data/utr/utr_aliases.csv
```

Outputs are written to `artifacts/`:

- `model.joblib`
- `metrics.json`
- `model_benchmarks.csv`
- `selected_model.json`
- `feature_importances.csv`
- `player_snapshot.csv`
- `player_live_state.csv`
- `pair_history.csv`
- `calibration_buckets.csv`
- `error_slices.csv`
- `test_predictions.csv`

## Refresh Elo / player state without full retraining

If you add fresher ATP yearly match files and want to update the player snapshot and pair history first:

```bash
python scripts/build_live_state.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_plus_stats_2026_refresh
```

That creates refreshed:

- `player_snapshot.csv`
- `player_live_state.csv`
- `pair_history.csv`

You can then point the main draw and live update workflows at those files with `--snapshot-path`, `--live-state-path`, and `--pair-history-path`.

## Leakage-safe hyperparameter tuning

You can now run a strict time-split hyperparameter search on the validation years without leaking test-season information:

```bash
python scripts/tune_hyperparameters.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_tuning \
  --validation-start-year 2024 \
  --test-start-year 2025
```

That writes:

- `hyperparameter_search_results.csv`
- `best_hyperparameters.json`

The tuner currently searches:

- histogram gradient boosting
- random forest
- XGBoost when your environment supports it

## Leakage-safe Elo tuning

To test which Elo `K` and decay settings work best on future-season data, run:

```bash
python scripts/tune_elo.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_elo_tuning \
  --validation-start-year 2024 \
  --test-start-year 2025
```

That writes:

- `elo_search_results.csv`
- `best_elo_config.json`
- `elo_config_only.json`

If you want to retrain the model using the tuned Elo settings:

```bash
python scripts/train_model.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_plus_stats_2026_model \
  --validation-start-year 2024 \
  --test-start-year 2025 \
  --elo-config-json artifacts_elo_tuning/elo_config_only.json
```

And if you want to rebuild the standalone live-state snapshot with the same Elo config:

```bash
python scripts/build_live_state.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_plus_stats_2026_refresh \
  --elo-config-json artifacts_elo_tuning/elo_config_only.json
```

## Predict a single match

```bash
python scripts/predict_match.py \
  --model-path artifacts/model.joblib \
  --snapshot-path artifacts/player_snapshot.csv \
  --player-1 "Carlos Alcaraz" \
  --player-2 "Jannik Sinner" \
  --match-date 2025-01-26 \
  --surface Hard \
  --best-of 5 \
  --round F \
  --tourney-level G \
  --draw-size 128
```

## Simulate a tournament

Create a first-round bracket CSV:

```csv
slot,player_1,player_2
1,Carlos Alcaraz,Qualifier A
2,Jannik Sinner,Qualifier B
3,Novak Djokovic,Qualifier C
4,Daniil Medvedev,Qualifier D
```

Then run:

```bash
python scripts/simulate_tournament.py \
  --model-path artifacts/model.joblib \
  --snapshot-path artifacts/player_snapshot.csv \
  --bracket-csv data/example_bracket.csv \
  --tournament-date 2025-01-13 \
  --surface Hard \
  --best-of 5 \
  --tourney-level G \
  --draw-size 8 \
  --simulations 10000
```

This writes player advancement probabilities to `artifacts/bracket_probabilities.csv`.

## Automated ATP draw workflow

You can now fetch an official ATP draw page directly, save the bracket, score round one, and simulate the tournament from that source.

Fetch only:

```bash
python scripts/fetch_atp_draw.py \
  --draw-url https://www.atptour.com/en/scores/current/miami/403/draws \
  --match-type qualifiersingles \
  --output-dir artifacts/miami_qual_draw
```

Full workflow:

```bash
python scripts/run_atp_tournament_workflow.py \
  --draw-url https://www.atptour.com/en/scores/current/miami/403/draws \
  --match-type qualifiersingles \
  --surface Hard \
  --output-dir artifacts/miami_qual_live \
  --simulations 10000
```

For a main draw, use the main draw page instead:

```bash
python scripts/run_atp_tournament_workflow.py \
  --draw-url https://www.atptour.com/en/scores/current/miami/403/draws \
  --surface Hard \
  --output-dir artifacts/miami_main_draw \
  --simulations 10000
```

The workflow writes:

- `bracket.csv`
- `draw_metadata.json`
- `first_round_predictions.csv`
- `tournament_probabilities.csv`

Qualifying draws are supported too. The simulator will label the terminal outcome as `Qualified` instead of `Champion` when you use `--match-type qualifiersingles`.

## Automated ATP Live Update

You can now point the project at an official ATP draw page and have it automatically:

1. fetch the draw
2. derive the ATP results page
3. scrape completed singles matches
4. pull the official ATP match stats for each completed match
5. update the live player state
6. re-score the current round
7. re-simulate the remaining tournament many times

Example for Miami qualifying:

```bash
python scripts/auto_update_tournament.py \
  --draw-url https://www.atptour.com/en/scores/current/miami/403/draws \
  --match-type qualifiersingles \
  --surface Hard \
  --output-dir artifacts_live/miami_auto_update \
  --simulations 5000
```

Example for a main draw:

```bash
python scripts/auto_update_tournament.py \
  --draw-url https://www.atptour.com/en/scores/current/miami/403/draws \
  --surface Hard \
  --output-dir artifacts_live/miami_main_auto_update \
  --simulations 5000
```

The automatic updater writes:

- `bracket.csv`
- `draw_metadata.json`
- `completed_results_auto.csv`
- `updated_live_state.csv`
- `updated_player_snapshot.csv`
- `updated_pair_history.csv`
- `completed_match_audit.csv`
- `current_round_bracket.csv`
- `current_round_predictions.csv`
- `remaining_tournament_probabilities.csv`
- `auto_update_summary.json`

`completed_results_auto.csv` is already shaped for the live feedback loop, including optional serve/return stat columns when ATP exposes them on the Stats Centre page. That means your rolling serve and return features get refreshed automatically as completed matches come in.

The updater also maintains a reusable long-run prediction log in `artifacts_plus_stats/`:

- `prediction_history.csv`
- `prediction_history_buckets.csv`
- `prediction_history_summary.json`

These files let you track questions like "when the model said 70%, how often was it actually right?"

## Live Feedback Loop

The project can now update player ratings and rolling form after completed matches, then re-score the current round and re-simulate the rest of the tournament.

First, build the richer live state artifact:

```bash
python scripts/build_live_state.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_plus_stats
```

That creates:

- `artifacts_plus_stats/player_live_state.csv`
- `artifacts_plus_stats/player_snapshot.csv`
- `artifacts_plus_stats/pair_history.csv`

Then create a completed results CSV using `data/completed_results_template.csv`. The only required columns are:

- `player_1`
- `player_2`
- `winner`
- `round_name`

`match_date`, `surface`, `best_of`, `tourney_level`, rankings, and serve/return stats are optional. If you include the optional stat columns, the rolling serve/return features will update too.

Run the feedback loop:

```bash
python scripts/run_live_feedback_loop.py \
  --model-path artifacts_plus_stats/model.joblib \
  --live-state-path artifacts_plus_stats/player_live_state.csv \
  --draw-metadata-json artifacts_live/miami_qualifying_20260316/draw_metadata.json \
  --completed-results-csv your_completed_results.csv \
  --output-dir artifacts_live/miami_feedback_update \
  --simulations 5000
```

The feedback loop writes:

- `updated_live_state.csv`
- `updated_player_snapshot.csv`
- `updated_pair_history.csv`
- `completed_match_audit.csv`
- `current_round_bracket.csv`
- `current_round_predictions.csv`
- `remaining_tournament_probabilities.csv`
- `feedback_summary.json`

It also updates the same central prediction-history files in `artifacts_plus_stats/` so you can evaluate calibration and accuracy over time across many tournaments.

## Ablation Experiments

You can now run a leakage-safe experiment suite across multiple feature sets:

```bash
python scripts/run_experiments.py \
  --data-dir data/raw/tennis_atp \
  --output-dir artifacts_experiments \
  --validation-start-year 2024 \
  --test-start-year 2025
```

This writes:

- `all_experiment_benchmarks.csv`
- `experiment_summary.csv`
- one benchmark CSV per feature set
- one calibration bucket CSV per feature set
- one error-slice CSV per feature set
- one test-prediction CSV per feature set

The intended loop is:

1. Fetch the official draw once.
2. Save completed matches into a small CSV as results come in.
3. Re-run `run_live_feedback_loop.py`.
4. Read the updated current-round predictions and remaining tournament probabilities.

## Miami Dashboard

Launch the local app:

```bash
streamlit run streamlit_app.py
```

The dashboard includes:

- a Miami contender board
- featured possible Miami matchups
- a manual match predictor with charts
- live match-sheet upload for qualifying or daily order-of-play style predictions
- bracket upload for full tournament simulation
- an official ATP draw importer that can fetch, score, and simulate directly from the ATP draw URL

For a Miami-style draw with byes, upload a full bracket CSV that includes `BYE` entries in the empty slots.
For qualifying, use the match-sheet template at `data/miami_open_2026_match_sheet_template.csv`.

## Current feature set

The upgraded pipeline uses pre-match features such as:

- ranking gap
- ranking points gap
- age and height gap
- overall win-rate gap
- surface-specific win-rate gap
- recent-form gap
- overall Elo gap
- recent Elo gap
- surface Elo gap
- best-of-5 Elo gap
- best-of-context Elo gap
- best-of-5 experience and win-rate gap
- days-since-last-match gap
- recent workload gap over the last 30 days
- serve-win-rate gap
- return-win-rate gap
- first-serve-in and first-serve-win gap
- second-serve-win gap
- ace-rate and double-fault-rate gap
- break-point-save gap
- recent serve/return gap
- surface serve/return gap
- UTR gap when historical UTR is supplied
- experience gap
- tournament context like surface, round, level, and best-of

## Optional UTR data

The project can merge a historical UTR feed onto ATP matches by taking the latest known UTR on or before each match date.

Required `utr_history.csv` columns:

```csv
player_name,rating_date,utr_singles
Carlos Alcaraz,2024-01-15,15.91
Jannik Sinner,2024-01-15,15.97
```

Optional `utr_aliases.csv` columns:

```csv
source_name,canonical_name
Alexander Zverev,Sascha Zverev
```

Use the alias file when the UTR player name and Jeff Sackmann player name do not line up exactly.

For weekly current-file refreshes, `utr_current.csv` can be as small as:

```csv
player_name,utr_singles
Carlos Alcaraz,16.12
Jannik Sinner,16.36
```

## Current benchmark result

On the current ATP dataset in this workspace, the selected model is histogram gradient boosting and the saved benchmark run produced:

- validation accuracy `0.6495`
- validation AUC `0.7156`
- validation log loss `0.6156`
- test accuracy `0.6593`
- test AUC `0.7252`
- test log loss `0.6081`
- calibrated test accuracy `0.6609`
- calibrated test AUC `0.7263`
- calibrated test log loss `0.6075`

## Important note

The next high-value improvements are opponent-strength-adjusted rolling stats, explicit head-to-head features, and more detailed serve/return metrics computed strictly from prior matches only.
