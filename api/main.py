from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Premier League Predictions API",
    description="API for serving ML-based Premier League match predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data/predictions/predictions_latest.parquet"
METADATA_PATH = PROJECT_ROOT / "data/metadata/fetch_state.json"


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
    """
    Get the latest gameweek predictions.
    
    Returns:
        - gameweek: The gameweek number
        - predictions: List of match predictions with probabilities
    """
    try:
        if not PREDICTIONS_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="No predictions available. Scheduler may not have run yet."
            )
        
        # Load predictions
        df_predictions = pd.read_parquet(PREDICTIONS_PATH)
        
        if df_predictions.empty:
            raise HTTPException(
                status_code=404,
                detail="Predictions file is empty."
            )
        
        # Convert to list of dicts
        predictions = df_predictions.to_dict('records')
        
        # Convert timestamps to ISO format strings
        for pred in predictions:
            if 'date' in pred and pd.notna(pred['date']):
                pred['date'] = pd.Timestamp(pred['date']).isoformat()
        
        # Try to get gameweek from upcoming matches file
        gameweek = None
        upcoming_path = PROJECT_ROOT / "data/upcoming/upcoming_gw_matches.parquet"
        if upcoming_path.exists():
            df_upcoming = pd.read_parquet(upcoming_path)
            if not df_upcoming.empty and 'matchweek' in df_upcoming.columns:
                gameweek = int(df_upcoming['matchweek'].iloc[0])
        
        return {
            "gameweek": gameweek,
            "count": len(predictions),
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)