Premier League Predictor
Overview

The Premier League Predictor is a data-driven project that predicts match outcomes for the English Premier League. The system collects historical and current season data, processes it into features, and will eventually train machine learning models to predict probabilities for Win, Draw, or Loss, as well as expected scorelines.

High-Level Pipeline
Match data + team statistics
        ↓
Feature engineering (pre-match only)
        ↓
Machine learning model (expected goals)
        ↓
Poisson probability model
        ↓
Scoreline probabilities
        ↓
Win / Draw / Loss probabilities


Key Idea:
Predict how many goals each team is expected to score, then derive all match outcomes from that.

Tech Stack
Layer	Technology / Library
Frontend	React
Backend / API	FastAPI
Data Fetching	Football Data API (via Python requests)
Data Storage	Parquet, Pandas
ML Models	XGBoost / LightGBM, Poisson