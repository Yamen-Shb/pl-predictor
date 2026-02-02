import pandas as pd
from datetime import date, timedelta
from data_collection import utils, metadata
import os

def fetch_matches_by_date(competition_id, date_from, date_to):
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "status": "FINISHED"
    }
    return utils.fetch_matches(competition_id, params)

def main():
    season = 2025
    competition_id = "PL"
    competition_key = f"pl_{season}"
    RAW_PATH = f"data/raw/pl_{season}.parquet"
    FLAT_PATH = f"data/processed/matches_flat_{season}.parquet"

    # load last fetched date
    metadataDate = metadata.load_metadata()
    date_from = metadataDate.get(competition_key, {}).get("last_fetched_date")

    if date_from is None:
        raise ValueError("No last_fetched_date found. Run historical loader first.")

    date_to = (date.today() - timedelta(days=1)).isoformat()

    matches = fetch_matches_by_date(competition_id, date_from, date_to)

    if not matches:
        print("No matches returned from API.")
        return

    if not os.path.exists(RAW_PATH) or not os.path.exists(FLAT_PATH):
        raise FileNotFoundError(
            "Raw or flat match files not found. Run historical_loader first."
        )
    # load existing raw data
    df_existing = pd.read_parquet(RAW_PATH)
    existing_ids = set(df_existing["id"])

    new_matches = [m for m in matches if m["id"] not in existing_ids]

    if not new_matches:
        print("No new matches to append.")
        return

    # append raw
    df_new = pd.DataFrame(new_matches)
    df_raw = pd.concat([df_existing, df_new], ignore_index=True)
    df_raw.to_parquet(RAW_PATH, index=False)

    # append flattened
    df_flat_new = utils.flatten_raw_match(df_new)
    df_flat_existing = pd.read_parquet(FLAT_PATH)
    df_flat = (
        pd.concat([df_flat_existing, df_flat_new], ignore_index=True)
        .sort_values("date")
    )

    df_flat.to_parquet(FLAT_PATH, index=False)

    # update metadata ONLY after success
    max_date = df_flat_new["date"].max().date().isoformat()
    metadata.update_last_fetched_date(competition_key, max_date)

    print(f"Weekly update complete. Updated through {max_date}")

if __name__ == "__main__":
    main()
