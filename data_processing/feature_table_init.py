import os
import pandas as pd

FEATURES_PATH = "data/features/features.parquet"

def init_features_table():
    columns = [
        "match_id",
        "date",
        "matchweek",
        "home_team",
        "away_team",
        "home_points_last5",
        "away_points_last5",
        "home_goals_for_last5",
        "away_goals_for_last5",
        "home_goals_against_last5",
        "away_goals_against_last5",
        "home_home_goals_for_avg",
        "home_home_goals_against_avg",
        "away_away_goals_for_avg",
        "away_away_goals_against_avg",
        "is_big_game",
        "h2h_home_points_last5",
        "h2h_away_points_last5",
        "home_max_goals_last5",
        "home_min_goals_last5",
        "home_scored_3plus_last5",
        "home_scored_0_last5",
        "home_clean_sheets_last5",
        "home_conceded_3plus_last5",
        "away_max_goals_last5",
        "away_min_goals_last5",
        "away_scored_3plus_last5",
        "away_scored_0_last5",
        "away_clean_sheets_last5",
        "away_conceded_3plus_last5",
        "home_goals",
        "away_goals"
    ]

    if os.path.exists(FEATURES_PATH):
        df = pd.read_parquet(FEATURES_PATH)
        print(f"Features table already exists with {len(df)} rows")
    else:
        df = pd.DataFrame(columns=columns)
        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        df.to_parquet(FEATURES_PATH, index=False)
        print(f"Initialized empty features table at {FEATURES_PATH}")

    return df

init_features_table()