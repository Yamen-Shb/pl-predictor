import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY or not API_KEY.strip():
    raise ValueError(
        "FOOTBALL_DATA_API_KEY is not set. Add it to .env or the environment."
    )
BASE_URL = "https://api.football-data.org/v4"
HEADERS = { 'X-Auth-Token': API_KEY }

def fetch_matches(competition_id, params):
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch matches. Status code: {response.status_code}"
        )

    data = response.json()
    return data.get("matches", [])

def save_raw_matches(matches, season):
    df = pd.DataFrame(matches)
    os.makedirs("data/raw", exist_ok=True)
    df.to_parquet(f"data/raw/pl_{season}.parquet", index=False)
    print(f"Saved {len(df)} matches to data/raw/pl_{season}.parquet")

def flatten_raw_match(df_raw):
    df = pd.DataFrame()
    df['match_id'] = df_raw['id']
    df['date'] = pd.to_datetime(df_raw['utcDate'])
    df['matchweek'] = df_raw['matchday']
    df['home_team'] = df_raw['homeTeam'].apply(lambda x: x['name'])
    df['away_team'] = df_raw['awayTeam'].apply(lambda x: x['name'])
    df['home_score'] = df_raw['score'].apply(lambda x: x.get('fullTime', {}).get('home') if x else None)
    df['away_score'] = df_raw['score'].apply(lambda x: x.get('fullTime', {}).get('away') if x else None)
    df['winner'] = df_raw['score'].apply(lambda x: x.get('winner') if x else None)

    df = df.sort_values('date').reset_index(drop=True)
    return df

def save_processed(df, season):
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(f"data/processed/matches_flat_{season}.parquet", index=False)
    print(f"Saved {len(df)} matches to data/processed/matches_flat_{season}.parquet")