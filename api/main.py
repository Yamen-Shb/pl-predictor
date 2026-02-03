from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data/predictions/predictions_latest.parquet"
UPCOMING_PATH = PROJECT_ROOT / "data/upcoming/upcoming_gw_matches.parquet"
METADATA_PATH = PROJECT_ROOT / "data/metadata/fetch_state.json"

predictions_df = None
upcoming_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictions_df, upcoming_df
    try:
        if PREDICTIONS_PATH.exists():
            predictions_df = pd.read_parquet(PREDICTIONS_PATH)
            logger.info(f"Loaded predictions: {len(predictions_df)} rows")
        else:
            logger.warning("Predictions file not found")

        if UPCOMING_PATH.exists():
            upcoming_df = pd.read_parquet(UPCOMING_PATH)
            logger.info(f"Loaded upcoming matches: {len(upcoming_df)} rows")
        else:
            logger.warning("Upcoming matches file not found")

    except Exception as e:
        logger.error(f"Failed to load data on startup: {e}")

    yield  # <-- this is where FastAPI runs until shutdown

    # Optional: cleanup code here if needed
    logger.info("Shutting down FastAPI lifespan")

app = FastAPI(
    title="Premier League Predictions API",
    description="API for serving ML-based Premier League match predictions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Premier League Predictions API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predictions": "/predictions/latest"
        }
    }

@app.get("/health")
async def health_check():
    """
    Returns 200 with status "healthy" when OK, or "degraded" when predictions
    or metadata are missing. Clients should treat "degraded" as unhealthy for
    readiness checks if they require predictions to be available.
    """
    try:
        # Check if prediction file exists
        predictions_exist = PREDICTIONS_PATH.exists()
        
        # Load metadata
        metadata = {}
        last_updated = None
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            
            # Get last upcoming match date as proxy for last update
            last_updated = metadata.get("pl_2025", {}).get("last_upcoming_match_date")
        
        return {
            "status": "healthy",
            "predictions_available": predictions_exist,
            "last_updated": last_updated,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/predictions/latest")
async def get_latest_predictions():
    if predictions_df is None or predictions_df.empty:
        raise HTTPException(
            status_code=404,
            detail="No predictions available. Scheduler may not have run yet."
        )

    predictions = predictions_df.to_dict("records")

    for pred in predictions:
        if "date" in pred and pd.notna(pred["date"]):
            pred["date"] = pd.Timestamp(pred["date"]).isoformat()

    gameweek = None
    if upcoming_df is not None and "matchweek" in upcoming_df.columns:
        gameweek = int(upcoming_df["matchweek"].iloc[0])

    return {
        "gameweek": gameweek,
        "count": len(predictions),
        "predictions": predictions,
        "generated_at": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)