import os
import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import timedelta
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_collection.metadata import load_metadata, save_metadata
from data_collection.upcoming_fetcher import main as run_fetch_upcoming
from data_processing.build_upcoming_features import build_upcoming_features
from models.predict_upcoming import predict_upcoming, load_models
from data_collection.weekly_fetcher import main as run_weekly_fetcher
from data_processing.feature_extraction import extract_and_append_features
from data_collection.metadata import get_last_upcoming_match_date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/metadata"
METADATA_PATH = DATA_DIR / "fetch_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def last_processed_match_date():
    metadata = load_metadata()
    last_date = metadata.get("pl_2025", {}).get("last_fetched_date")
    if last_date:
        # Metadata stores date-only strings; treat them as UTC dates to avoid
        # tz-naive vs tz-aware comparison issues.
        ts = pd.to_datetime(last_date)
        if isinstance(ts, pd.Timestamp) and ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts
    return None


def newest_match_date():
    match_files = [
        PROJECT_ROOT / "data/processed/matches_flat_2023.parquet",
        PROJECT_ROOT / "data/processed/matches_flat_2024.parquet",
        PROJECT_ROOT / "data/processed/matches_flat_2025.parquet",
    ]

    dfs = []
    for f in match_files:
        if f.exists():
            dfs.append(pd.read_parquet(f))
    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    latest = df_all["date"].max()
    # Normalize to UTC-aware timestamp for consistent comparisons.
    if isinstance(latest, pd.Timestamp):
        if latest.tz is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")
    return latest


def run_weekly_pipeline() -> bool:
    """
    Run weekly fetcher, feature extraction, and training.
    Returns True if new data was processed, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("WEEKLY PIPELINE - Checking for new completed matches")
    logger.info("=" * 60)

    # Ensure project root is on path and cwd so relative imports/paths work
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    data_collection_dir = PROJECT_ROOT / "data_collection"
    if str(data_collection_dir) not in sys.path:
        sys.path.insert(0, str(data_collection_dir))

    # Check last processed date
    last_date = last_processed_match_date()

    # 1. Weekly fetcher
    try:
        logger.info("Running weekly fetcher...")
        run_weekly_fetcher()
    except Exception as e:
        logger.error(f"Weekly fetcher failed: {e}")
        return False
    
    latest_match = newest_match_date()

    if last_date:
        logger.info(f"Last processed match date: {last_date.date()}")
    if latest_match:
        logger.info(f"Latest match in data: {latest_match.date()}")
    else:
        logger.warning("No match data found. Skipping weekly pipeline.")
        return False

    if last_date and latest_match.date() <= last_date.date():
        logger.info("No new matches found. Skipping feature extraction and training.")
        return False

    # 2. Feature extraction
    try:
        logger.info("Loading flattened match data for feature extraction...")
        df_2023 = pd.read_parquet(PROJECT_ROOT / "data/processed/matches_flat_2023.parquet")
        df_2024 = pd.read_parquet(PROJECT_ROOT / "data/processed/matches_flat_2024.parquet")
        df_2025 = pd.read_parquet(PROJECT_ROOT / "data/processed/matches_flat_2025.parquet")
        df_all = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)

        logger.info("Running feature extraction...")
        extract_and_append_features(
            df_all,
            features_path=str(PROJECT_ROOT / "data/features/features.parquet"),
            start_season=2023,
            start_matchweek=6
        )
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return False

    # 3. Training
    try:
        from models.train import main as run_train
        logger.info("Training models...")
        run_train(show_plots=False)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

    logger.info("Weekly pipeline complete. Updating metadata...")

    metadata = load_metadata()
    if "pl_2025" not in metadata:
        metadata["pl_2025"] = {}
    metadata["pl_2025"]["last_fetched_date"] = str(latest_match.date())
    save_metadata(metadata)
    
    logger.info(f"Metadata updated with last_fetched_date: {latest_match.date()}")
    return True


def run_upcoming_pipeline() -> bool:
    """
    Run upcoming fetcher, feature extraction, and predictions.
    Returns True if upcoming matches were processed, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("UPCOMING PIPELINE - Checking for next GW matches")
    logger.info("=" * 60)

    # Ensure project root is on path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    # Get current time
    now = pd.Timestamp.now(tz='UTC')
    
    last_match_str = get_last_upcoming_match_date("pl_2025")
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

    # 1. Fetch upcoming matches (handles cutoff logic internally)
    try:
        logger.info("Running upcoming fetcher...")
        run_fetch_upcoming()
    except Exception as e:
        logger.error(f"Upcoming fetcher failed: {e}")
        return False
    
    # Check if we have upcoming matches to predict
    upcoming_path = PROJECT_ROOT / "data/upcoming/upcoming_gw_matches.parquet"
    if not upcoming_path.exists():
        logger.info("No upcoming matches file found. Skipping predictions.")
        return False

    df_upcoming = pd.read_parquet(upcoming_path)
    if df_upcoming.empty:
        logger.info("No upcoming matches to predict. Skipping.")
        return False

    logger.info(f"Found {len(df_upcoming)} upcoming matches.")

    # 2. Extract features for upcoming matches
    try:
        logger.info("Building features for upcoming matches...")
        df_features = build_upcoming_features()
        
        if df_features.empty:
            logger.error("Feature extraction returned empty DataFrame.")
            return False
    except Exception as e:
        logger.error(f"Upcoming feature extraction failed: {e}")
        return False

    # 3. Generate predictions
    try:
        logger.info("Loading models...")
        home_model, away_model = load_models()
        
        logger.info("Generating predictions...")
        predictions = predict_upcoming(
            df_features,
            home_model=home_model,
            away_model=away_model
        )
        
        # Save predictions
        predictions_dir = PROJECT_ROOT / "data/predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / "predictions_latest.parquet"
        
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_parquet(predictions_path, index=False)
        
        logger.info(f"Saved {len(predictions)} predictions to {predictions_path}")
        
        # Display summary
        gw = df_features["matchweek"].iloc[0] if "matchweek" in df_features.columns else "Unknown"
        logger.info(f"\n{'='*60}")
        logger.info(f"GW {gw} Predictions Summary")
        logger.info(f"{'='*60}")
        for pred in predictions:
            home = pred.get('home_team', 'Home')
            away = pred.get('away_team', 'Away')
            h_pred = pred['home_pred']
            a_pred = pred['away_pred']
            h_win = pred['home_win_pct'] * 100
            draw = pred['draw_pct'] * 100
            a_win = pred['away_win_pct'] * 100
            
            logger.info(
                f"{home} {h_pred} - {a_pred} {away} | "
                f"H:{h_win:.1f}% D:{draw:.1f}% A:{a_win:.1f}%"
            )
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        return False

    logger.info("Upcoming pipeline complete.")
    return True


def run_daily_pipeline() -> None:
    """
    Main daily scheduler that runs both weekly and upcoming pipelines.
    """
    logger.info("=" * 60)
    logger.info("DAILY SCHEDULER STARTED")
    logger.info("=" * 60)

    # Run weekly pipeline (completed matches)
    weekly_ran = run_weekly_pipeline()

    # Run upcoming pipeline (future matches)
    upcoming_ran = run_upcoming_pipeline()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DAILY SCHEDULER SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Weekly pipeline: {'✓ Completed' if weekly_ran else '○ Skipped (no new data)'}")
    logger.info(f"Upcoming pipeline: {'✓ Completed' if upcoming_ran else '○ Skipped (no matches or before cutoff)'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_daily_pipeline()