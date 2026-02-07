import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# Columns dropped for model input 
ID_COLUMNS = ["match_id", "matchweek", "date", "home_team", "away_team", "home_goals", "away_goals"]


def load_models():
    home_path = ARTIFACTS_DIR / "xgb_home_goals.json"
    away_path = ARTIFACTS_DIR / "xgb_away_goals.json"
    if not home_path.exists() or not away_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run models/train.py first."
        )
    home_model = xgb.XGBRegressor()
    home_model.load_model(str(home_path))
    away_model = xgb.XGBRegressor()
    away_model.load_model(str(away_path))
    logging.info("Loaded home and away XGBoost models from artifacts.")
    return home_model, away_model


def get_feature_columns(df: pd.DataFrame) -> list:
    """Columns to use as model input (exclude ids and targets)."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def predict_upcoming(
    upcoming_df: pd.DataFrame,
    home_model: xgb.XGBRegressor | None = None,
    away_model: xgb.XGBRegressor | None = None,
) -> list[dict]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.poisson_layer import poisson_matrix, match_outcome_probabilities

    if home_model is None or away_model is None:
        home_model, away_model = load_models()

    feature_cols = get_feature_columns(upcoming_df)
    if not feature_cols:
        raise ValueError(
            "upcoming_df must contain feature columns (e.g. from feature extraction)."
        )

    X = upcoming_df[feature_cols].copy()
    X = X.fillna(0)

    home_preds = home_model.predict(X)
    away_preds = away_model.predict(X)

    results = []
    logging.info(f"Running predictions for {len(upcoming_df)} upcoming matches...")

    for i, row in upcoming_df.reset_index(drop=True).iterrows():
        home_pred = float(home_preds[i])
        away_pred = float(away_preds[i])

        score_matrix = poisson_matrix(home_pred, away_pred)
        home_win_pct, draw_pct, away_win_pct = match_outcome_probabilities(score_matrix)

        out = {
            "home_pred": int(round(home_pred)),  # Rounded to nearest integer
            "away_pred": int(round(away_pred)),  # Rounded to nearest integer
            "home_win_pct": round(float(home_win_pct), 4),
            "draw_pct": round(float(draw_pct), 4),
            "away_win_pct": round(float(away_win_pct), 4),
        }

        for key in ["match_id", "date", "home_team", "away_team"]:
            if key in row.index and pd.notna(row.get(key)):
                out[key] = row[key]

        results.append(out)

    logging.info("Predictions complete.")
    return results
