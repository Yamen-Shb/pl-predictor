import optuna
import pandas as pd
import xgboost as xgb
from pathlib import Path
import json
import pickle

# Import from your train.py
from train import (
    temporal_train_val_test_split,
    split_features_targets,
    evaluate_match_predictions
)

FEATURES_PATH = "data/features/features.parquet"
ARTIFACTS_DIR = Path("models/artifacts")

def objective(trial):
    """Optuna objective function"""
    
    # Load data
    df = pd.read_parquet(FEATURES_PATH)
    train_df, val_df, _ = temporal_train_val_test_split(df)

    objective_name = trial.suggest_categorical("objective", ["count:poisson", "reg:tweedie"])
    
    # Suggest hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),

        "objective": objective_name,
        "early_stopping_rounds": 30,
        "random_state": 42,
    }
    
    # Objective-specific settings
    if objective_name == "count:poisson":
        params["eval_metric"] = "poisson-nloglik"
    else:
        params["eval_metric"] = "rmse"
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.80)

    # Optional: tune Dixon–Coles rho as well (helps draw probs)
    dixon_coles_rho = trial.suggest_float("dixon_coles_rho", -0.20, -0.03)
    
    # Train home model
    X_train_home, y_train_home = split_features_targets(train_df, "home_goals")
    X_val_home, y_val_home = split_features_targets(val_df, "home_goals")
    
    home_model = xgb.XGBRegressor(**params)
    home_model.fit(
        X_train_home, y_train_home,
        eval_set=[(X_train_home, y_train_home), (X_val_home, y_val_home)],
        verbose=False
    )
    
    # Train away model
    X_train_away, y_train_away = split_features_targets(train_df, "away_goals")
    X_val_away, y_val_away = split_features_targets(val_df, "away_goals")
    
    away_model = xgb.XGBRegressor(**params)
    away_model.fit(
        X_train_away, y_train_away,
        eval_set=[(X_train_away, y_train_away), (X_val_away, y_val_away)],
        verbose=False
    )
    
    # Evaluate match-level metrics
    metrics = evaluate_match_predictions(
        home_model, away_model,
        X_val_home, y_val_home,
        X_val_away, y_val_away,
        label="Validation",
        dixon_coles_rho=dixon_coles_rho
    )
    
    # Outcome accuracy (most important for predictions)
    return -metrics['outcome_accuracy_pct']
    


def main():
    print("Starting Optuna hyperparameter tuning...")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='xgboost_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=300,  # Try 300 different combinations
        show_progress_bar=True,
        n_jobs=1  # Use 1 job to avoid conflicts
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best outcome accuracy: {-study.best_value:.2f}%")
    print(f"\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
   # Save best params (Option A structure)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    best_params_path = ARTIFACTS_DIR / "best_params.json"

    best = dict(study.best_params)
    rho = best.pop("dixon_coles_rho", None)

    payload = {
        "xgb_params": best,           # only params that go into XGBRegressor
        "dixon_coles_rho": rho        # poisson-layer param
    }

    with open(best_params_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved best parameters to {best_params_path}")

    with open(ARTIFACTS_DIR / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

if __name__ == "__main__":
    main()