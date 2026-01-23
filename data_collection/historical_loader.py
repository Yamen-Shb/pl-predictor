import pandas as pd
import utils

def fetch_matches_for_season(season, competition_id):
    params = {
        "season": season,
        "status": "FINISHED"
    }
    return utils.fetch_matches(competition_id, params)

seasons = [2023, 2024, 2025]
for season in seasons:
    matches = fetch_matches_for_season(season, 'PL')
    utils.save_raw_matches(matches, season)
    df_raw = pd.read_parquet(f"data/raw/pl_{season}.parquet")
    df_flat = utils.flatten_raw_match(df_raw)
    utils.save_processed(df_flat, season)