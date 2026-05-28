import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

def get_spark_session():
    import pyspark
    spark_version = pyspark.__version__
    
    return SparkSession.builder \
        .appName("StockDataStreaming") \
        .config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_2.12:{spark_version}") \
        .getOrCreate()

def run_streaming(broker, checkpoint_dir, output_dir):
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Define the schema that matches our JSON producer payload
    schema = StructType([
        StructField("timestamp", StringType(), True),
        StructField("ticker", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("volume", IntegerType(), True)
    ])

    print(f"Connecting to Kafka broker: {broker}")
    
    # Read stream from all stock_ topics
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", broker) \
        .option("subscribePattern", "stock_.*") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Parse the JSON from the "value" column
    parsed_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Ensure output directories exist (PySpark will create them, but good practice if local)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Writing to Parquet at: {output_dir}")
    print(f"Using Checkpoint Dir: {checkpoint_dir}")
    
    # Write the stream to Parquet files
    # Partitioning by ticker is useful for downstream ML processing
    query = parsed_df.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", output_dir) \
        .option("checkpointLocation", checkpoint_dir) \
        .partitionBy("ticker") \
        .start()

    print("Streaming started. Waiting for termination...")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("Streaming stopped by user.")
        query.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Structured Streaming for Stock Data")
    parser.add_argument('--broker', type=str, default='localhost:9092', help='Kafka broker address')
    args = parser.parse_args()
    
    # Define paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_location = os.path.join(base_dir, "data", "checkpoints")
    output_location = os.path.join(base_dir, "data", "parquet_files")
    
    run_streaming(args.broker, checkpoint_location, output_location)