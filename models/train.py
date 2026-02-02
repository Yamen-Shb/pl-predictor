import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt

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
        columns=["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    y = df[target_col]
    return X, y


# model training
def train_xgboost_regressor(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,        
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    return model


# evaluation metrics
def evaluate_model(model, X, y, label):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    print(f"{label} MAE:  {mae:.3f}")
    print(f"{label} RMSE: {rmse:.3f}")
    return mae, rmse


# feature gain & learning curve
def plot_model_diagnostics(model, X_train, y_train, X_val, y_val, title_prefix="Model"):
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
    if results:
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
        plt.plot(x_axis, results['validation_1']['rmse'], label='Val')
        plt.xlabel('Boosting Round')
        plt.ylabel('RMSE')
        plt.title(f"{title_prefix} - Learning Curve")
        plt.legend()
        plt.show()


# main training pipeline
def main():
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

    plot_model_diagnostics(home_model, X_train, y_train, X_val, y_val, "Home Goals")

    home_model.save_model(ARTIFACTS_DIR / "xgb_home_goals.json")

    print("\nTraining AWAY goals model...")
    X_train, y_train = split_features_targets(train_df, "away_goals")
    X_val, y_val = split_features_targets(val_df, "away_goals")
    X_test, y_test = split_features_targets(test_df, "away_goals")

    away_model = train_xgboost_regressor(X_train, y_train, X_val, y_val)

    print("\nAway goals evaluation:")
    evaluate_model(away_model, X_val, y_val, "Val")
    evaluate_model(away_model, X_test, y_test, "Test")

    plot_model_diagnostics(away_model, X_train, y_train, X_val, y_val, "Away Goals")

    away_model.save_model(ARTIFACTS_DIR / "xgb_away_goals.json")

    print("\nTraining complete. Models saved.")


if __name__ == "__main__":
    main()