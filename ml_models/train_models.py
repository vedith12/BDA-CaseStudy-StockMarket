import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from feature_engineering import engineer_features
from ensemble_model import build_voting_ensemble, build_stacking_ensemble

def train_all_models(data_dir, model_dir):
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    try:
        df = pd.read_parquet(data_dir)
    except Exception as e:
        print(f"Error loading Parquet data: {e}")
        return
        
    if df.empty:
        print("Dataframe is empty.")
        return

    print("Engineering features...")
    df = engineer_features(df)
    
    # Define features
    features = ['price', 'volume', 'lag1', 'lag2', 'lag3', 'MA5', 'MA10', 'MA20', 
                'rolling_mean', 'rolling_std', 'price_change', 'pct_change']
                
    unique_tickers = df['ticker'].unique()
    print(f"Found {len(unique_tickers)} tickers: {unique_tickers}")
    
    for ticker in unique_tickers:
        print(f"\n--- Training for {ticker} ---")
        df_ticker = df[df['ticker'] == ticker].copy()
        
        if df_ticker.empty or len(df_ticker) < 10:
            print(f"Not enough data for {ticker}. Skipping.")
            continue
            
        X = df_ticker[features]
        y = df_ticker['target']
        
        # We do a time-series split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Initialize individual models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1)),
            'XGBoost': XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
        }
        
        trained_estimators = []
        
        # Train base models
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            # Evaluate
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            # Accuracy heuristic
            accuracy = max(0, 100 - (mae / (y_test.mean() or 1)) * 100)
            
            print(f"   {name} -> MAE: {mae:.4f}, Est. Accuracy: {accuracy:.2f}%")
            
            # Save model with ticker prefix
            filename = f"{ticker}_" + name.lower().replace(" ", "_") + "_model.joblib"
            path = os.path.join(model_dir, filename)
            joblib.dump(model, path)
            
            # Add to estimators list for ensemble
            trained_estimators.append((name.lower().replace(" ", "_"), model))

        # Train Voting Ensemble
        voting = build_voting_ensemble(trained_estimators)
        voting.fit(X_train, y_train)
        vote_preds = voting.predict(X_test)
        vote_accuracy = max(0, 100 - (mean_absolute_error(y_test, vote_preds) / (y_test.mean() or 1)) * 100)
        print(f"   Voting Ensemble -> MAE: {mean_absolute_error(y_test, vote_preds):.4f}, Acc: {vote_accuracy:.2f}%")
        joblib.dump(voting, os.path.join(model_dir, f'{ticker}_voting_ensemble_model.joblib'))
        
        # Train Stacking Ensemble
        stacking = build_stacking_ensemble(trained_estimators)
        stacking.fit(X_train, y_train)
        stack_preds = stacking.predict(X_test)
        stack_accuracy = max(0, 100 - (mean_absolute_error(y_test, stack_preds) / (y_test.mean() or 1)) * 100)
        print(f"   Stacking Ensemble -> MAE: {mean_absolute_error(y_test, stack_preds):.4f}, Acc: {stack_accuracy:.2f}%")
        joblib.dump(stacking, os.path.join(model_dir, f'{ticker}_ensemble_model.joblib')) # Main ensemble
        
    print(f"\nAll specialized models successfully saved to {model_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "parquet_files")
    model_path = os.path.join(base_dir, "models", "saved_models")
    
    train_all_models(data_path, model_path)