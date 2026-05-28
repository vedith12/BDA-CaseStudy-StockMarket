import os
import glob
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import json

# Add ml_models to path so we can import feature_engineering and train_models if needed
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_models_dir = os.path.join(base_dir, "ml_models")
sys.path.append(ml_models_dir)
from train_models import train_all_models
from feature_engineering import engineer_features

app = FastAPI(title="Stock Prediction System API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for models
loaded_models = {}

def load_models_into_memory():
    """Loads all joblib models from disk into memory."""
    model_dir = os.path.join(base_dir, "models", "saved_models")
    if not os.path.exists(model_dir):
        return
        
    for filename in os.listdir(model_dir):
        if filename.endswith('.joblib'):
            parts = filename.replace('_model.joblib', '').split('_', 1)
            if len(parts) == 2:
                ticker = parts[0].upper()
                model_name = parts[1].replace('_', ' ').title()
                
                # Normalize ensemble names globally
                if model_name == "Ensemble": model_name = "Stacking Ensemble"
                
                key = f"{ticker}_{model_name}"
                model_path = os.path.join(model_dir, filename)
                try:
                    loaded_models[key] = joblib.load(model_path)
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models_into_memory()

# Schemas
class FeatureInput(BaseModel):
    price: float
    volume: int
    lag1: float
    lag2: float
    lag3: float
    MA5: float
    MA10: float
    MA20: float
    rolling_mean: float
    rolling_std: float
    price_change: float
    pct_change: float

class PredictRequest(BaseModel):
    ticker: str
    features: FeatureInput
    model_name: str = "Stacking Ensemble"

@app.get("/stream-data")
def get_stream_data(ticker: str = "AAPL", mode: str = "historical"):
    """
    Simulates fetching latest data from our storage (Parquet files).
    In a real app, this might query an in-memory DB or MongoDB.
    """
    parquet_dir = os.path.join(base_dir, "data", "parquet_files")
    
    # Check if files exist
    if not os.path.exists(parquet_dir):
        return []
        
    try:
        # Read ONLY the specific ticker's partition using pushdown filters. 
        # This prevents loading the entire massive dataset into memory.
        df_ticker = pd.read_parquet(parquet_dir, filters=[('ticker', '=', ticker)])
        
        if df_ticker.empty:
            return []
            
        df_ticker = df_ticker.sort_values(by='timestamp')
        
        # We MUST engineer the features dynamically right now so the frontend doesn't get
        # empty sets (NaN/undefined) for critical data points like Lag and Moving Averages
        try:
            df_ticker = engineer_features(df_ticker)
            # Drop duplicates (prevents endless looping data when the market is closed)
            df_ticker = df_ticker.drop_duplicates(subset=['timestamp'])
        except Exception as fe_error:
            print(f"Warning during feature engineering in stream-data: {fe_error}")
            pass
            
        # Depending on mode, limit the amount of data we return
        # Here we just return the last 100 rows for the frontend chart visualization
        df_ticker = df_ticker.fillna(0)
        records = df_ticker.tail(100).to_dict(orient="records")
        return records
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return []
        
@app.post("/train-model")
async def train_model_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers the retraining of our ML models in the background.
    """
    data_path = os.path.join(base_dir, "data", "parquet_files")
    model_path = os.path.join(base_dir, "models", "saved_models")
    
    background_tasks.add_task(train_all_models, data_path, model_path)
    # Background tasks don't reload gracefully, so we'd need another endpoint to trigger reload
    return {"message": "Model training started in background."}

@app.post("/reload-models")
def reload_models():
    """Reloads models from disk into memory."""
    load_models_into_memory()
    return {"message": f"Loaded {len(loaded_models)} models."}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Predict next price using specifically selected model.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded.")
        
    model_key = f"{req.ticker.upper()}_{req.model_name}"
    
    if model_key not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found. Ensure {req.ticker} is trained.")
        
    model = loaded_models[model_key]
    
    # Format incoming data for the model
    data = pd.DataFrame([req.features.dict()])
    
    # Expected feature order needs to match training
    feature_order = ['price', 'volume', 'lag1', 'lag2', 'lag3', 'MA5', 'MA10', 'MA20', 
                     'rolling_mean', 'rolling_std', 'price_change', 'pct_change']
    data = data[feature_order]
    
    try:
        prediction = model.predict(data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Fake confidence interval based on basic heuristics
    interval = 0.02
    if "Ensemble" in req.model_name:
        interval = 0.01 # Ensembles have tighter confidence bounds loosely speaking :)
        
    return {
        "model_used": req.model_name,
        "predicted_price": round(float(prediction), 2),
        "confidence_lower": round(float(prediction) * (1 - interval), 2),
        "confidence_upper": round(float(prediction) * (1 + interval), 2)
    }

@app.get("/model-metrics")
def get_metrics(ticker: str = "AAPL"):
    """Returns actual metrics for available models from metrics.json."""
    metrics_path = os.path.join(base_dir, "models", "saved_models", "metrics.json")
    
    if not os.path.exists(metrics_path):
        return {
            "ticker": ticker.upper(),
            "metrics": {
                "Stacking Ensemble": "N/A",
                "Random Forest": "N/A",
                "XGBoost": "N/A",
                "Voting Ensemble": "N/A",
                "Linear Regression": "N/A",
                "SVR": "N/A"
            }
        }
        
    try:
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
            
        ticker_metrics = all_metrics.get(ticker.upper(), {})
        
        return {
            "ticker": ticker.upper(),
            "metrics": ticker_metrics
        }
    except Exception as e:
        print(f"Error reading metrics.json: {e}")
        return {"error": "Failed to load metrics"}

@app.get("/feature-importance")
def get_feature_importance():
    """
    Extracts feature importances from the loaded Random Forest or XGBoost model.
    """
    model_keys = list(loaded_models.keys())
    
    # Try RF first, then XGB
    target_model_key = None
    if "Random Forest" in model_keys:
        target_model_key = "Random Forest"
    elif "Xgboost" in model_keys:
        target_model_key = "Xgboost"
        
    if not target_model_key:
        return {"error": "No tree-based model found to extract importances."}
        
    model = loaded_models[target_model_key]
    
    feature_names = ['price', 'volume', 'lag1', 'lag2', 'lag3', 'MA5', 'MA10', 'MA20', 
                     'rolling_mean', 'rolling_std', 'price_change', 'pct_change']
                     
    importances = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        
    if not importances:
        return {"error": "Model does not support feature_importances_"}
        
    # Group and sort
    results = [{"feature": f, "importance": round(imp, 4)} for f, imp in zip(feature_names, importances)]
    results = sorted(results, key=lambda x: x['importance'], reverse=True)
    return results