from collections import Counter
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from pathlib import Path
import json

FEATURES_PATH = "data/features/features.parquet"
ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# temporal split
def temporal_train_val_test_split(df, train_frac=0.7, val_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# drop identifiers and separate target
def split_features_targets(df, target_col):
    X = df.drop(
        columns=["match_id", "matchweek", "date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    y = df[target_col]
    return X, y

def load_best_params():
    """Load best params from Optuna tuning if available"""
    best_params_path = ARTIFACTS_DIR / "best_params.json"
    
    default_xgb = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.5,
        "reg_lambda": 1.28,
        "objective": "count:poisson",
    }
    default_rho = -0.10

    if not best_params_path.exists():
        print("Using default hyperparameters")
        return default_xgb, default_rho

    with open(best_params_path, "r") as f:
        payload = json.load(f)

    # Support both formats:
    # - new: {"xgb_params": {...}, "dixon_coles_rho": ...}
    # - old: flat dict from older runs
    if "xgb_params" in payload:
        xgb_params = payload.get("xgb_params") or {}
        rho = float(payload.get("dixon_coles_rho", default_rho))
    else:
        xgb_params = dict(payload)
        rho = float(xgb_params.pop("dixon_coles_rho", default_rho))

    # fill any missing defaults
    merged = dict(default_xgb)
    merged.update(xgb_params)

    return merged, rho


# model training
def train_xgboost_regressor(X_train, y_train, X_val, y_val):
    params, _ = load_best_params()

    objective = params.pop("objective", "count:poisson")

    params.update({
        "objective": objective,
        "early_stopping_rounds": 30,
        "random_state": 42,
    })

    if objective == "count:poisson":
        params["eval_metric"] = ["poisson-nloglik"]
    elif objective == "reg:tweedie":
        params["eval_metric"] = ["rmse"]
        params["tweedie_variance_power"] = float(params.get("tweedie_variance_power", 1.3))

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    return model

# combined evaluation for both models (match-level metrics)
def evaluate_match_predictions(home_model, away_model, X_home, y_home, X_away, y_away, label, dixon_coles_rho=-0.10):
    """
    Evaluate exact score accuracy and match outcome accuracy using Poisson probabilities
    """
    from poisson_layer import poisson_matrix, match_outcome_probabilities
    import numpy as np
    
    home_preds = home_model.predict(X_home)
    away_preds = away_model.predict(X_away)
    
    y_home_actual = y_home.values
    y_away_actual = y_away.values
    
    # Use most likely scoreline from Poisson distribution
    most_likely_home = []
    most_likely_away = []
    pred_outcomes = []
    
    for h_exp, a_exp in zip(home_preds, away_preds):
        # Get Poisson probability matrix
        score_matrix = poisson_matrix(h_exp, a_exp, dixon_coles_rho=dixon_coles_rho)  # Adjust rho based on validation tuning
        
        # Find the most likely scoreline for exact score prediction
        max_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
        h_most_likely = max_idx[0]
        a_most_likely = max_idx[1]
        
        most_likely_home.append(h_most_likely)
        most_likely_away.append(a_most_likely)
        
        # Outcome: use AGGREGATED probabilities with smart draw prediction
        home_win_prob, draw_prob, away_win_prob = match_outcome_probabilities(score_matrix)

        # Softer draw prediction: allow draw if it's competitive
        max_prob = max(home_win_prob, draw_prob, away_win_prob)
        
        if max_prob == draw_prob:
            pred_outcomes.append('D')
        elif home_win_prob > away_win_prob:
            pred_outcomes.append('H')
        else:
            pred_outcomes.append('A')
    
    most_likely_home = np.array(most_likely_home)
    most_likely_away = np.array(most_likely_away)
    
    # Exact score accuracy
    exact_scores = (most_likely_home == y_home_actual) & (most_likely_away == y_away_actual)
    exact_score_pct = exact_scores.mean() * 100
    
    # Actual outcomes
    def get_outcome(home, away):
        if home > away:
            return 'H'
        elif home < away:
            return 'A'
        else:
            return 'D'
    
    actual_outcomes = [get_outcome(h, a) for h, a in zip(y_home_actual, y_away_actual)]
    
    correct_outcomes = sum(p == a for p, a in zip(pred_outcomes, actual_outcomes))
    outcome_accuracy_pct = (correct_outcomes / len(pred_outcomes)) * 100
    
    # Distribution of predictions vs actuals
    pred_outcome_dist = Counter(pred_outcomes)
    actual_outcome_dist = Counter(actual_outcomes)
    
    print(f"\n{label} Match-Level Metrics:")
    print(f"{'='*50}")
    print(f"Exact Score:             {exact_score_pct:.1f}%")
    print(f"Correct Outcome (H/D/A): {outcome_accuracy_pct:.1f}%")
    print(f"\nPredicted Outcomes: H={pred_outcome_dist.get('H', 0)} D={pred_outcome_dist.get('D', 0)} A={pred_outcome_dist.get('A', 0)}")
    print(f"Actual Outcomes:    H={actual_outcome_dist.get('H', 0)} D={actual_outcome_dist.get('D', 0)} A={actual_outcome_dist.get('A', 0)}")
    
    return {
        'exact_score_pct': exact_score_pct,
        'outcome_accuracy_pct': outcome_accuracy_pct,
        'pred_outcome_dist': pred_outcome_dist,
        'actual_outcome_dist': actual_outcome_dist
    }

# evaluation metrics
def evaluate_model(model, X, y, label):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    print(f"{label} MAE:  {mae:.3f}")
    print(f"{label} RMSE: {rmse:.3f}")
    return mae, rmse


# feature gain & learning curve
def plot_model_diagnostics(model, X_train, y_train, X_val, y_val, title_prefix="Model", show_plots=True):
    if not show_plots:
        return
    import matplotlib.pyplot as plt
    
    booster = model.get_booster()
    
    feature_names = list(X_train.columns)
    
    plt.figure(figsize=(12, 8))
    
    # get feature importance as a dictionary
    importance_dict = model.get_booster().get_score(importance_type='gain')
    
    # map feature names
    mapped_importance = {}
    for i, fname in enumerate(feature_names):
        fkey = f'f{i}'
        if fkey in importance_dict:
            mapped_importance[fname] = importance_dict[fkey]
    
    if mapped_importance:
        sorted_features = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        features, gains = zip(*sorted_features)
        
        plt.barh(range(len(features)), gains)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Gain')
        plt.title(f"{title_prefix} - Top 20 Feature Gain")
        plt.gca().invert_yaxis()
        plt.tight_layout()
    plt.show()

    # Learning curves
    results = model.evals_result()
    if results and 'validation_0' in results:
        # Determine which metric was used (could be poisson-nloglik or rmse)
        available_metrics = list(results['validation_0'].keys())
        
        if available_metrics:
            metric_name = available_metrics[0]  # use the first available metric
            epochs = len(results['validation_0'][metric_name])
            x_axis = range(epochs)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(x_axis, results['validation_0'][metric_name], label='Train')
            if 'validation_1' in results:
                plt.plot(x_axis, results['validation_1'][metric_name], label='Val')
            plt.xlabel('Boosting Round')
            plt.ylabel(metric_name)
            plt.title(f"{title_prefix} - {metric_name}")
            plt.legend()

            plt.tight_layout()
            plt.show()


# main training pipeline
def main(show_plots):
    if not Path(FEATURES_PATH).exists():
        raise FileNotFoundError(
            f"Features file not found at {FEATURES_PATH}. Run feature extraction first."
        )
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)


    train_df, val_df, test_df = temporal_train_val_test_split(df)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

    print("\nTraining HOME goals model...")
    X_train, y_train = split_features_targets(train_df, "home_goals")
    X_val, y_val = split_features_targets(val_df, "home_goals")
    X_test, y_test = split_features_targets(test_df, "home_goals")

    home_model = train_xgboost_regressor(X_train, y_train, X_val, y_val)

    print("\nHome goals evaluation:")
    evaluate_model(home_model, X_val, y_val, "Val")
    evaluate_model(home_model, X_test, y_test, "Test")

    plot_model_diagnostics(home_model, X_train, y_train, X_val, y_val, "Home Goals", show_plots)

    home_model.save_model(ARTIFACTS_DIR / "xgb_home_goals.json")

    print("\nTraining AWAY goals model...")
    X_train, y_train = split_features_targets(train_df, "away_goals")
    X_val, y_val = split_features_targets(val_df, "away_goals")
    X_test, y_test = split_features_targets(test_df, "away_goals")

    away_model = train_xgboost_regressor(X_train, y_train, X_val, y_val)

    print("\nAway goals evaluation:")
    evaluate_model(away_model, X_val, y_val, "Val")
    evaluate_model(away_model, X_test, y_test, "Test")

    plot_model_diagnostics(away_model, X_train, y_train, X_val, y_val, "Away Goals", show_plots)

    away_model.save_model(ARTIFACTS_DIR / "xgb_away_goals.json")

    print("\nTraining complete. Models saved.")


if __name__ == "__main__":
    main(show_plots=True)