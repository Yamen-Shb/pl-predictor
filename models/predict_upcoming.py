import pandas as pd
from pathlib import Path
import logging
import sys
from models.predict_logic import predict_upcoming, load_models

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_processing.build_upcoming_features import build_upcoming_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("data")
UPCOMING_DIR = DATA_DIR / "upcoming"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"

UPCOMING_PATH = UPCOMING_DIR / "upcoming_gw_matches.parquet"
UPCOMING_FEATURES_PATH = FEATURES_DIR / "upcoming_features.parquet"
PREDICTIONS_PATH = PREDICTIONS_DIR / "predictions_latest.parquet"


def main():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load upcoming matches
    if not UPCOMING_PATH.exists():
        logging.error(f"No upcoming matches found at {UPCOMING_PATH}")
        logging.info("Run fetch_upcoming.py first.")
        return
    
    df_upcoming = pd.read_parquet(UPCOMING_PATH)
    
    if df_upcoming.empty:
        logging.warning("Upcoming matches file is empty. Nothing to predict.")
        return
    
    logging.info(f"Loaded {len(df_upcoming)} upcoming matches for prediction.")
    
    # Extract features for upcoming matches
    logging.info("Building features for upcoming matches...")
    df_features = build_upcoming_features()
    
    if df_features.empty:
        logging.error("Feature extraction returned empty DataFrame.")
        return
    
    # Load models once
    home_model, away_model = load_models()
    
    # Generate predictions
    predictions = predict_upcoming(
        df_features,
        home_model=home_model,
        away_model=away_model
    )
    
    # Convert to DataFrame and save
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_parquet(PREDICTIONS_PATH, index=False)
    
    logging.info(f"Saved predictions to {PREDICTIONS_PATH}")
    
    # Display summary
    if not df_predictions.empty:
        gw = df_features["matchweek"].iloc[0] if "matchweek" in df_features.columns else "Unknown"
        logging.info(f"\n{'='*60}")
        logging.info(f"GW {gw} Predictions Summary")
        logging.info(f"{'='*60}")
        for _, row in df_predictions.iterrows():
            home = row.get('home_team', 'Home')
            away = row.get('away_team', 'Away')
            h_pred = row['home_pred']
            a_pred = row['away_pred']
            h_win = row['home_win_pct'] * 100
            draw = row['draw_pct'] * 100
            a_win = row['away_win_pct'] * 100
            
            logging.info(
                f"{home} {h_pred} - {a_pred} {away} | "
                f"H:{h_win:.1f}% D:{draw:.1f}% A:{a_win:.1f}%"
            )


if __name__ == "__main__":
    main()