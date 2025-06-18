# Databricks notebook source
!pip install sdv

# COMMAND ----------

# MAGIC %run /Workspace/Users/ballam@gitam.in/project_mimic/mlops/src/pre_processing/Storage_connections

# COMMAND ----------

df=spark.read.format("delta").load(f"{processed_data_path}/df_cleaned")

# COMMAND ----------

df.display

# COMMAND ----------

from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import pandas as pd

# Step 1: Read Delta table
df_spark = df.toPandas()

# Step 2: Convert to Pandas
df = pd.DataFrame(df_spark.dropna())

# Step 3: Automatically detect metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# Step 4: Create and train the model
model = SingleTablePreset(metadata=metadata, name='FAST_ML')
model.fit(df)

# Step 5: Generate synthetic data
synthetic_data = model.sample(num_rows=1000)

# Step 6: Display result
print(synthetic_data.head())

# COMMAND ----------

synthetic_data.display()

# COMMAND ----------

synthetic_data.info()

# COMMAND ----------

from pyspark.sql import SparkSession

# Step 1: Convert synthetic_data (Pandas) → Spark DataFrame
synthetic_spark_df = spark.createDataFrame(synthetic_data)

# Step 2: Define path to store in DBFS
temp_dbfs_path = "dbfs:/tmp/synthetic_ctgan_data"

# Step 3: Save as Delta
synthetic_spark_df.write.mode("overwrite").format("delta").save(temp_dbfs_path)

# Step 4: Create temporary view (accessible in SQL / Spark)
synthetic_spark_df.createOrReplaceTempView("temp_synthetic_data")

print("✅ Synthetic data saved in DBFS and registered as temp view 'temp_synthetic_data'")