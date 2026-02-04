# Premier League Match Predictor

A machine learning system that predicts Premier League match scores using XGBoost regression models. The system automatically fetches match data, extracts features, trains models, and generates predictions for upcoming gameweeks.

## What It Does

- **Predicts match scores** for upcoming Premier League fixtures
- **Calculates win probabilities** (Home Win / Draw / Away Win)
- **Updates automatically** via daily scheduled jobs
- **Tracks team form** using rolling statistics and historical data
- **Serves predictions** through a FastAPI backend and React frontend

## Tech Stack

**Backend & ML**
- **Python 3.x** - Core language
- **XGBoost** - Gradient boosting for regression
- **Poisson Layer** - Outcome Probabilities
- **pandas** - Data manipulation
- **scikit-learn** - Model evaluation
- **FastAPI** - REST API server
- **Render** - Backend deployment

**Frontend**
- **React** - User interface
- **Vercel** - Frontend deployment

**Data Source**
- **Football-Data.org API** - Match data and fixtures

## How It Works

### 1. Data Collection

**Historical Loader** (for completed matches):
- Fetches finished matches from Football-Data.org API
- Flattens and stores match data by season

**Weekly Pipeline** (for completed matches):
- Fetches finished matches from Football-Data.org API
- Flattens and stores match data by season
- Only processes new finished matches since last saved finished match date

**Upcoming Pipeline** (for future matches):
- Fetches next gameweek fixtures
- Runs after current gameweek ends

### 2. Feature Engineering

For each match, the system extracts **25+ features** including:

**Recent Form (Last 5 Matches)**
- Points earned
- Goals scored and conceded
- Max/min goals in a match
- Count of 3+ goal performances
- Count of scoreless matches
- Clean sheets kept

**Venue-Specific Stats**
- Average goals scored at home (for home team)
- Average goals conceded at home
- Average goals scored away (for away team)
- Average goals conceded away

**Head-to-Head**
- Points earned in last 5 H2H meetings
- Historical rivalry indicator

**Match Context**
- Big game indicator (derbies, top-6 clashes)
- Matchweek number

### 3. Model Training

**Architecture:**
- Two separate XGBoost regressors (one for home goals, one for away goals)
- Temporal train/validation/test split (70/15/15)
- Early stopping to prevent overfitting

**Key Hyperparameters:**
```python
n_estimators=500
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8,
reg_alpha=1.5  # L1 regularization
reg_lambda=1.28  # L2 regularization
objective="reg:squarederror"
eval_metric=["rmse", "mae"]
```

**Why Two Models?**
- Home and away goal distributions differ
- Allows model to learn venue-specific patterns
- More accurate than single multi-output model

### 4. Prediction Generation

For each upcoming match:
1. Extract features using all historical data up to prediction time
2. Predict home goals and away goals using trained models
3. Round predictions to nearest 0.5 for display
4. Calculate win probabilities using Poisson distribution assumption

**Win Probability Calculation:**
```
P(Home Win) = P(home_goals > away_goals)
P(Draw) = P(home_goals == away_goals)
P(Away Win) = P(home_goals < away_goals)
```

### 5. Automated Scheduling

The `scheduler.py` runs daily and:
1. **Checks for new completed matches** → Triggers weekly pipeline if found
2. **Checks for upcoming fixtures** → Triggers upcoming pipeline if within cutoff
3. **Updates metadata** to track last processed dates
4. **Logs all operations** for monitoring

## 🚀 Setup & Usage

### Prerequisites
```bash
pip install requirements.txt
```

### Environment Variables
Create a `.env` file:
```
FOOTBALL_DATA_API_KEY=your_api_key_here
```

### Running Locally

**1. Fetch Historical Data**
```bash
python data_collection/historical_loader.py
```

**2. Build Feature Table**
```bash
python data_processing/feature_table_init.py
```

**3. Extract Features**
```bash
python data_processing/feature_extraction.py
```

**4. Train Models**
```bash
python models/train.py
```

**5. Generate Predictions**
```bash
python scheduler.py
```

### Deployment

**Backend (Render):**
- Deploy FastAPI app

**GitHub Workflow:**
- Give daily-pipline.yml the necessary permissions
- Scheduled pipeline runs scheduler.py once a day

**Frontend (Vercel):**
- Deploy React app
- Configure API endpoint to Render backend

## Model Performance

Typical metrics on test set:
- **MAE**: ~0.9-1.0 goals
- **RMSE**: ~1.1-1.2 goals

The model predicts the correct exact score ~13% of the time and correct match outcome ~50% of the time.

## Key Design Decisions

**Why rolling windows?**
- Recent form (last 5 matches) is more predictive than season-long averages
- Captures momentum and current team state

**Why temporal splits?**
- Prevents data leakage (no future information in training)
- Realistic evaluation of deployment performance

**Why separate home/away models?**
- Home advantage is real and significant (~0.3-0.5 goal difference)
- Models learn different patterns for each venue

**Why XGBoost?**
- Handles non-linear relationships between features
- Built-in regularization prevents overfitting
- Fast training and prediction
- Interpretable feature importance

## License

This project is for educational purposes. Match data sourced from Football-Data.org under their terms of use.

## 🙏 Acknowledgments

- Football-Data.org for providing free API access
- XGBoost developers for the excellent ML library
- The Premier League for being endlessly unpredictable! 

---

**Note:** This model is for entertainment and educational purposes only. Always gamble responsibly if using predictions for betting.