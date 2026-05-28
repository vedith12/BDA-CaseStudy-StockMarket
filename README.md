# 📈 End-to-End Stock Market Analysis & Prediction System

This project is a complete Big Data and Machine Learning pipeline that streams, stores, processes, and predicts live stock market prices.

---

## 🏗️ System Architecture (Flow of Events)

1. **Apache Kafka (The Ingestion Layer)**
   - Our `kafka_producer.py` continuously pulls live stock data (AAPL, TSLA, INFY) from Yahoo Finance.
   - It acts as a "Publisher," streaming these thousands of data points at high speed into a Kafka Broker topic without dropping packets.
2. **Apache Spark (The Processing Layer)**
   - `spark_stream.py` acts as the "Subscriber." It constantly listens to Kafka in the background.
   - It grabs those raw data strings, organizes them, and formally writes them into highly compressed data lakes called **Parquet** files inside our `/data` folder.
3. **Machine Learning (The Brain)**
   - `train_models_no_spark.py` reads those massive Parquet files locally.
   - It engineers heavy mathematical features (Moving Averages, Lag, Volatility).
   - It trains **Individualized Machine Learning Ensembles** per stock without Spark (using `scikit-learn` and `xgboost`), saves the brains in `/models`, and importantly, saves the real-time accuracy percentages into `metrics.json` for the frontend to display.
4. **FastAPI (The Bridge)**
   - The backend `main.py` constantly hosts the trained ML models in memory.
   - When requested, it executes the predictive math in milliseconds and serves formatted JSON data natively to the web.
5. **React.js Dashboard (The UI)**
   - Our front-end hits the API every 5 seconds to grab the latest streams, calculate market closure states, and visually graph the actual vs. expected predictions in a beautiful interactive chart.

---

### Step 1: Start the Big Data Core (Zookeeper & Kafka)
*Note: Make sure your Kafka directory paths match where you installed it.*
```bash
# Terminal 1: Zookeeper

cd home/vanam/kafka_2.13-3.7.0
bin/zookeeper-server-start.sh config/zookeeper.properties

# Terminal 2: Kafka
cd home/vanam/kafka_2.13-3.7.0
bin/kafka-server-start.sh config/server.properties
```

### Step 2: Start the Data Pipeline
```bash
# Terminal 3: Spark Streaming (Saves data to Parquet endlessly)
cd /mnt/e/BDA-kafkaCS/stock_prediction_system/data_pipeline
env -u SPARK_HOME python3 spark_stream.py

# Terminal 4: Kafka Realtime Producer (Pulls live data)
cd /mnt/e/BDA-kafkaCS/stock_prediction_system/data_pipeline
python3 kafka_producer.py --mode realtime --stocks AAPL TSLA GOOG INFY
```
```bash
# Terminal 5: ML Model Training & Accuracy Metrics
cd /mnt/e/BDA-kafkaCS/stock_prediction_system/ml_models
while true; do python3 train_models_no_spark.py; echo "Sleeping for 1 Hour..."; sleep 3600; done 
```


### Step 3: Serve the API & Models
```bash
# Terminal 6: FastAPI Backend
cd /mnt/e/BDA-kafkaCS/stock_prediction_system/api
python3 -m uvicorn main:app --reload --port 8000
```

### Step 4: Boot the UI
```bash
# Terminal 7: React Frontend Dashboard
cd /mnt/e/BDA-kafkaCS/stock_prediction_system/frontend
npm run dev
```

Finally, click `http://localhost:5173/` in the React terminal to view your dashboard!
