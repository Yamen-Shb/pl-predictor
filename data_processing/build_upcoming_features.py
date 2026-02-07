import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_processing.feature_extraction import (
    compute_team_rolling_features,
    compute_venue_averages,
    compute_h2h_features,
    compute_is_big_game,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPCOMING_MATCHES_PATH = PROJECT_ROOT / "data/upcoming/upcoming_gw_matches.parquet"
FEATURES_PATH = PROJECT_ROOT / "data/features/features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/features/upcoming_features.parquet"


HISTORIC_TOP6 = [
    "Arsenal FC",
    "Chelsea FC",
    "Liverpool FC",
    "Manchester City FC",
    "Manchester United FC",
    "Tottenham Hotspur FC",
]

DERBY_GROUPS = [
    {"teams": ["Arsenal FC", "Chelsea FC", "Tottenham Hotspur FC"], "weight": 1.0},
    {"teams": ["Crystal Palace FC", "Arsenal FC", "Chelsea FC"], "weight": 0.5},
    {"teams": ["Liverpool FC", "Everton FC"], "weight": 1.0},
    {"teams": ["Manchester United FC", "Manchester City FC"], "weight": 1.0},
    {"teams": ["Newcastle United FC", "Sunderland AFC"], "weight": 0.8},
]

ROLLING_WINDOW = 5

def load_flat_matches_for_history() -> pd.DataFrame:
    flat_paths = [
        PROJECT_ROOT / "data/processed/matches_flat_2023.parquet",
        PROJECT_ROOT / "data/processed/matches_flat_2024.parquet",
        PROJECT_ROOT / "data/processed/matches_flat_2025.parquet",
    ]
    dfs = [pd.read_parquet(p) for p in flat_paths if p.exists()]
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)

def compute_features_for_upcoming_match(historical_data: pd.DataFrame, match: pd.Series) -> dict:
    home_team = match["home_team"]
    away_team = match["away_team"]

    home_points_last5, home_goals_for_last5, home_goals_against_last5, \
    home_max_goals, home_min_goals, home_scored_3plus, home_scored_0, \
    home_clean_sheets, home_conceded_3plus = compute_team_rolling_features(historical_data, home_team, ROLLING_WINDOW)

    away_points_last5, away_goals_for_last5, away_goals_against_last5, \
    away_max_goals, away_min_goals, away_scored_3plus, away_scored_0, \
    away_clean_sheets, away_conceded_3plus = compute_team_rolling_features(historical_data, away_team, ROLLING_WINDOW)

    home_home_goals_for_avg, home_home_goals_against_avg = \
        compute_venue_averages(historical_data, home_team, "home")

    away_away_goals_for_avg, away_away_goals_against_avg = \
        compute_venue_averages(historical_data, away_team, "away")

    h2h_home_points_last5, h2h_away_points_last5 = \
        compute_h2h_features(historical_data, home_team, away_team, ROLLING_WINDOW)

    is_big_game = compute_is_big_game(
        home_team,
        away_team,
        DERBY_GROUPS,
        HISTORIC_TOP6
    )

    return {
        "home_points_last5": home_points_last5,
        "away_points_last5": away_points_last5,
        "home_goals_for_last5": home_goals_for_last5,
        "away_goals_for_last5": away_goals_for_last5,
        "home_goals_against_last5": home_goals_against_last5,
        "away_goals_against_last5": away_goals_against_last5,
        "home_home_goals_for_avg": home_home_goals_for_avg,
        "home_home_goals_against_avg": home_home_goals_against_avg,
        "away_away_goals_for_avg": away_away_goals_for_avg,
        "away_away_goals_against_avg": away_away_goals_against_avg,
        "h2h_home_points_last5": h2h_home_points_last5,
        "h2h_away_points_last5": h2h_away_points_last5,
        "is_big_game": is_big_game,
        "home_max_goals_last5": home_max_goals,
        "home_min_goals_last5": home_min_goals,
        "home_scored_3plus_last5": home_scored_3plus,
        "home_scored_0_last5": home_scored_0,
        "home_clean_sheets_last5": home_clean_sheets,
        "home_conceded_3plus_last5": home_conceded_3plus,
        "away_max_goals_last5": away_max_goals,
        "away_min_goals_last5": away_min_goals,
        "away_scored_3plus_last5": away_scored_3plus,
        "away_scored_0_last5": away_scored_0,
        "away_clean_sheets_last5": away_clean_sheets,
        "away_conceded_3plus_last5": away_conceded_3plus
    }

def build_upcoming_features() -> pd.DataFrame:
    upcoming_df = pd.read_parquet(UPCOMING_MATCHES_PATH)
    historical_matches = load_flat_matches_for_history() 
    historical_matches["date"] = pd.to_datetime(historical_matches["date"])
    upcoming_df["date"] = pd.to_datetime(upcoming_df["date"])

    upcoming_features = []

    for _, match in tqdm(
        upcoming_df.iterrows(),
        total=len(upcoming_df),
        desc="Building upcoming features"
    ):
        historical_data = historical_matches[
            historical_matches["date"] < match["date"]
        ]

        if len(historical_data) == 0:
            logging.getLogger(__name__).warning(
                f"No history before {match['date']} for {match['home_team']} v {match['away_team']}; skipping."
            )
            continue

        computed = compute_features_for_upcoming_match(historical_data, match)

        row = {
            "match_id": match["match_id"],
            "date": match["date"],
            "matchweek": match["matchweek"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            **computed,
        }

        upcoming_features.append(row)

    df_upcoming_features = pd.DataFrame(upcoming_features)
    df_upcoming_features = df_upcoming_features.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_upcoming_features.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(df_upcoming_features)} upcoming feature rows → {OUTPUT_PATH}")

    return df_upcoming_features

if __name__ == "__main__":
    build_upcoming_features()