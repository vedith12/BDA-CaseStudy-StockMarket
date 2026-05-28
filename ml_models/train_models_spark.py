import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, lag, avg, stddev, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def get_spark_session():
    return SparkSession.builder \
        .appName("StockDataMLTraining") \
        .getOrCreate()

def engineer_features_spark(df):
    # Sort by timestamp
    window_spec = Window.partitionBy("ticker").orderBy("timestamp")
    
    # Target: Next price
    df = df.withColumn("target", lead("price", 1).over(window_spec))
    
    # Lags
    df = df.withColumn("lag1", lag("price", 1).over(window_spec))
    df = df.withColumn("lag2", lag("price", 2).over(window_spec))
    df = df.withColumn("lag3", lag("price", 3).over(window_spec))
    
    # Moving Averages
    window_5 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-4, 0)
    window_10 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-9, 0)
    window_20 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-19, 0)
    
    df = df.withColumn("MA5", avg("price").over(window_5))
    df = df.withColumn("MA10", avg("price").over(window_10))
    df = df.withColumn("MA20", avg("price").over(window_20))
    
    df = df.withColumn("rolling_mean", avg("price").over(window_20))
    df = df.withColumn("rolling_std", stddev("price").over(window_20))
    
    # Price changes
    df = df.withColumn("price_change", col("price") - col("lag1"))
    df = df.withColumn("pct_change", (col("price_change") / col("lag1")) * 100)
    
    # Drop rows with nulls (due to lag/lead and rolling windows)
    df = df.dropna()
    return df

def train_spark_models(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Loading data from {data_dir}...")
    try:
        df = spark.read.parquet(data_dir)
    except Exception as e:
        print(f"Error loading Parquet data: {e}")
        return
        
    if df.count() == 0:
        print("Dataframe is empty.")
        return
        
    print("Engineering features via Spark SQL...")
    df = engineer_features_spark(df)
    
    features_list = ['price', 'volume', 'lag1', 'lag2', 'lag3', 'MA5', 'MA10', 'MA20', 
                     'rolling_mean', 'rolling_std', 'price_change', 'pct_change']
                     
    assembler = VectorAssembler(inputCols=features_list, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    
    tickers = [row['ticker'] for row in df.select('ticker').distinct().collect()]
    print(f"Found {len(tickers)} tickers: {tickers}")
    
    all_metrics = {}
    evaluator_mae = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="mae")
    
    for ticker in tickers:
        print(f"\n--- Training for {ticker} (Spark ML) ---")
        df_ticker = df.filter(col("ticker") == ticker)
        
        # Train test split (80-20)
        train_df, test_df = df_ticker.randomSplit([0.8, 0.2], seed=42)
        
        if train_df.count() < 10:
            print(f"Not enough data for {ticker}. Skipping.")
            continue
            
        models = {
            'Linear Regression': LinearRegression(featuresCol="features", labelCol="target"),
            'Random Forest': RandomForestRegressor(featuresCol="features", labelCol="target", numTrees=50, maxDepth=10, seed=42),
            'GBTRegressor': GBTRegressor(featuresCol="features", labelCol="target", maxIter=50, seed=42)
        }
        
        ticker_metrics = {}
        
        test_mean_row = test_df.select(avg("target")).collect()
        test_mean = test_mean_row[0][0] if test_mean_row and test_mean_row[0][0] else 1.0
        
        for name, model in models.items():
            try:
                fitted_model = model.fit(train_df)
                predictions = fitted_model.transform(test_df)
                
                mae = evaluator_mae.evaluate(predictions)
                
                # Accuracy heuristic
                accuracy = max(0, 100 - (mae / test_mean) * 100)
                print(f"   {name} -> MAE: {mae:.4f}, Est. Accuracy: {accuracy:.2f}%")
                
                ticker_metrics[name] = f"{accuracy:.2f}%"
                
                # Save model directory (Spark saves as directories)
                filename = f"{ticker}_spark_" + name.lower().replace(" ", "_")
                path = os.path.join(model_dir, filename)
                fitted_model.write().overwrite().save(path)
            except Exception as e:
                print(f"   {name} -> Failed to train: {e}")
                
        all_metrics[ticker] = ticker_metrics
        
    print(f"\nAll Spark models successfully saved to {model_dir}")
    
    metrics_path = os.path.join(model_dir, "spark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Spark Metrics saved to {metrics_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "parquet_files")
    model_path = os.path.join(base_dir, "models", "saved_models")
    
    train_spark_models(data_path, model_path)
