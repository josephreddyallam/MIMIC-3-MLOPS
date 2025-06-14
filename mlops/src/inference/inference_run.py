import pandas as pd
from pyspark.sql import SparkSession
from inference_pipeline import load_model, predict, plot_roc_curve
from sklearn.metrics import classification_report, roc_auc_score

# Initialize Spark
spark = SparkSession.builder.getOrCreate()


spark_df = spark.read.format("delta").load("dbfs:/tmp/synthetic_ctgan_data")
df = spark_df.toPandas()

# Step 2: Load latest model from Azure
connection_string = ""
container_name = ""
model, expected_features = load_model(connection_string, container_name)

# Step 3: Run inference
preds, probs = predict(df, model)
df["predicted_mortality"] = preds
df["mortality_probability"] = probs

# Step 4: Evaluation
if "hospital_expire_flag" in df.columns:
    print("✅ Classification Report:")
    print(classification_report(df["hospital_expire_flag"], preds))

    roc_auc = roc_auc_score(df["hospital_expire_flag"], probs)
    print(f"✅ ROC AUC Score: {roc_auc:.2f}")
    plot_roc_curve(df["hospital_expire_flag"], probs)
else:
    print("⚠️ 'hospital_expire_flag' not found. Skipping evaluation.")

# Step 5: Fix column names for Delta (no special chars)
df.columns = df.columns.str.replace(r"[ ,;{}()\n\t=]", "_", regex=True)

# Step 6: Save inference output to Delta
spark_df = spark.createDataFrame(df)
spark_df.write.mode("overwrite").format("delta").save("dbfs:/tmp/inference_test_output")