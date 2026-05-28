import yfinance as yf
from kafka import KafkaProducer
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_kafka_producer(bootstrap_servers='localhost:9092'):
    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            api_version=(0, 10, 1) # Good default for many kafka broker versions
        )
        return producer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        return None

def fetch_and_stream(stock_ticker, mode, broker_address):
    producer = get_kafka_producer(broker_address)
    if not producer:
        return

    topic = f'stock_{stock_ticker}'
    logger.info(f"Starting to stream to topic: {topic} in {mode} mode")
    
    try:
        if mode == 'historical':
            # Fetch last 3 months of data at 5-minute intervals
            logger.info(f"Fetching historical data for {stock_ticker}...")
            data = yf.download(stock_ticker, period="60d", interval="5m")
            
            if data.empty:
                logger.error(f"No historical data found for {stock_ticker}")
                return
                
            logger.info(f"Fetched {len(data)} records. Starting streaming...")
            for index, row in data.iterrows():
                # Extract scalar values from potentially multi-indexed pandas objects
                # yfinance sometimes returns Series/DataFrames depending on how it's called
                price = float(row['Close']) if not isinstance(row['Close'], pd.Series) else float(row['Close'].iloc[0])
                volume = int(row['Volume']) if not isinstance(row['Volume'], pd.Series) else int(row['Volume'].iloc[0])
                
                payload = {
                    "timestamp": str(index),
                    "ticker": stock_ticker,
                    "price": price,
                    "volume": volume
                }
                
                producer.send(topic, value=payload)
                logger.info(f"Sent: {payload}")
                
                # Simulate streaming delay
                time.sleep(0.05) 
                
        elif mode == 'realtime':
            logger.info(f"Starting real-time streaming for {stock_ticker}. Press Ctrl+C to stop.")
            while True:
                ticker_data = yf.Ticker(stock_ticker)
                # Fetch 5 days to guarantee we get data even if today is a holiday or too early in the morning
                hist = ticker_data.history(period='5d', interval='1m')
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    current_vol = int(hist['Volume'].iloc[-1])
                    
                    payload = {
                        "timestamp": str(hist.index[-1]),
                        "ticker": stock_ticker,
                        "price": current_price,
                        "volume": current_vol
                    }
                    
                    producer.send(topic, value=payload)
                    logger.info(f"Sent real-time: {payload}")
                else:
                    logger.warning("No data retrieved in this tick.")
                    
                # Flush to ensure it's sent immediately
                producer.flush()
                # Wait before next fetch
                time.sleep(60) 
    
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user.")
    except Exception as e:
        logger.error(f"An error occurred during streaming: {e}")
    finally:
        producer.flush()
        producer.close()
        logger.info("Kafka producer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Data Kafka Producer")
    parser.add_argument('--stocks', nargs='+', default=['AAPL'], help='List of stock ticker symbols (e.g. AAPL TSLA GOOG)')
    parser.add_argument('--mode', type=str, choices=['historical', 'realtime'], default='historical', help='Streaming mode')
    parser.add_argument('--broker', type=str, default='localhost:9092', help='Kafka broker address')
    args = parser.parse_args()
    
    # Needs pandas for historical mode multi-index series parsing check
    import pandas as pd
    
    logger.info(f"Initiating streaming for {len(args.stocks)} stocks: {', '.join(args.stocks)}")
    
    with ThreadPoolExecutor(max_workers=len(args.stocks)) as executor:
        for stock in args.stocks:
            executor.submit(fetch_and_stream, stock.upper(), args.mode, args.broker)