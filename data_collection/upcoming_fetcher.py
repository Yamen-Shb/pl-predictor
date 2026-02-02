import pandas as pd
from datetime import timedelta
from pathlib import Path
from data_collection import utils, metadata

SEASON = 2025
COMPETITION_ID = "PL"
COMPETITION_KEY = f"pl_{SEASON}"

DATA_DIR = Path("data")
UPCOMING_DIR = DATA_DIR / "upcoming"

UPCOMING_PATH = UPCOMING_DIR / "upcoming_gw_matches.parquet"

def main():
    UPCOMING_DIR.mkdir(parents=True, exist_ok=True)

    # Get current time
    now = pd.Timestamp.now(tz='UTC')
    
    last_match_str = metadata.get_last_upcoming_match_date(COMPETITION_KEY)
    if last_match_str:
        # Ensure timezone-aware (UTC) - metadata strings may be naive
        last_match = pd.to_datetime(last_match_str)
        if last_match.tz is None:
            last_match = last_match.tz_localize('UTC')
        else:
            last_match = last_match.tz_convert('UTC')
    else:
        last_match = None
    
    if last_match is not None:
        # Only fetch after last match + 3 hours
        cutoff = last_match + timedelta(hours=3)
        if now < cutoff:
            print(f"⏸ Too soon. Last match at {last_match}. Next fetch after {cutoff}.")
            return
        date_from = last_match.strftime("%Y-%m-%d")
        filter_after = last_match
    else:
        # First run or no previous upcoming: fetch from today
        date_from = now.strftime("%Y-%m-%d")
        filter_after = None

    date_to = (now + timedelta(days=21)).strftime("%Y-%m-%d")
    
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "status": "SCHEDULED"
    }
    
    raw_matches = utils.fetch_matches(COMPETITION_ID, params)
    
    if not raw_matches:
        print("No upcoming scheduled matches.")
        # Create empty parquet if it doesn't exist
        pd.DataFrame().to_parquet(UPCOMING_PATH, index=False)
        return

    df_raw = pd.DataFrame(raw_matches)
    df_flat = utils.flatten_raw_match(df_raw)
    
    if filter_after is not None:
        # Ensure both sides are timezone-aware for comparison
        if df_flat["date"].dtype.tz is None:
            df_flat["date"] = df_flat["date"].dt.tz_localize('UTC')
        df_flat = df_flat[df_flat["date"] > filter_after]
    
    if df_flat.empty:
        print("No new matches after last fetch.")
        return
    
    # Get the next gameweek
    next_gw = df_flat["matchweek"].min()
    df_upcoming = df_flat[df_flat["matchweek"] == next_gw].copy()
    
    # Save upcoming matches (overwrites existing file)
    df_upcoming.sort_values("date").to_parquet(UPCOMING_PATH, index=False)
    
    # Update metadata with the last match datetime of this GW
    last_match_datetime = df_upcoming["date"].max()
    # Ensure timezone-aware (UTC) before saving
    if pd.isna(last_match_datetime):
        return
    if last_match_datetime.tz is None:
        last_match_datetime = last_match_datetime.tz_localize('UTC')
    else:
        last_match_datetime = last_match_datetime.tz_convert('UTC')
    # Store as ISO format string (includes timezone info)
    metadata.update_last_upcoming_match_date(COMPETITION_KEY, last_match_datetime.isoformat())
    
    date_range = f"{df_upcoming['date'].min().strftime('%Y-%m-%d %H:%M')} to {df_upcoming['date'].max().strftime('%Y-%m-%d %H:%M')}"
    print(
        f"✓ Fetched GW {next_gw} | "
        f"{len(df_upcoming)} matches | "
        f"{date_range}"
    )
    print(f"  Metadata updated: last_upcoming_match_date = {last_match_datetime}")

if __name__ == "__main__":
    main()