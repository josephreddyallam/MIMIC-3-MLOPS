# Databricks notebook source
# MAGIC %md
# MAGIC Required Libraraies

# COMMAND ----------

# MAGIC %pip install (
# MAGIC     pandas \
# MAGIC     numpy \
# MAGIC     scikit-learn \
# MAGIC     imbalanced-learn \
# MAGIC     matplotlib \
# MAGIC     seaborn \
# MAGIC     azure-storage-blob \
# MAGIC     joblib \
# MAGIC     xgboost \
# MAGIC     lightgbm \
# MAGIC     tqdm \
# MAGIC     sdv \
# MAGIC     optuna )

# COMMAND ----------

# MAGIC %md
# MAGIC Importing Files

# COMMAND ----------

# MAGIC %run /Workspace/Users/ballam@gitam.in/project_mimic/mlops/src/pre_processing/data_postprocessing.py 

# COMMAND ----------

# MAGIC %run /Workspace/Users/ballam@gitam.in/project_mimic/mlops/src/pre_processing/Storage_connections

# COMMAND ----------

# MAGIC %run /Workspace/Users/ballam@gitam.in/project_mimic/mlops/src/pre_processing/preprocessing

# COMMAND ----------

# MAGIC %run /Workspace/Users/ballam@gitam.in/project_mimic/mlops/src/training/model_train

# COMMAND ----------

# MAGIC
# MAGIC %run ./model_utils  

# COMMAND ----------

# MAGIC %run /Users/ballam@gitam.in/project_mimic/mlops/src/training/list_models_in_container.py

# COMMAND ----------

# MAGIC %run /Users/ballam@gitam.in/project_mimic/mlops/src/training/load_latest_model

# COMMAND ----------

# MAGIC %md
# MAGIC loading raw_data

# COMMAND ----------


df_raw = spark.read.format("delta").load(f"{raw_data_path}/df_final")

# COMMAND ----------

# MAGIC %md
# MAGIC Processing and storing raw_data

# COMMAND ----------

df_cleaned = preprocess_and_store_raw(df_raw, processed_data_path, spark)

# COMMAND ----------

# MAGIC %md
# MAGIC declaring column types for dataframe

# COMMAND ----------

import pandas as pd


df_pandas = df_cleaned.toPandas()


categorical_cols = df_pandas.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df_pandas.select_dtypes(include=['int64', 'float64']).columns.tolist()


binary_categoricals = [col for col in df_pandas.columns if df_pandas[col].nunique() == 2 and df_pandas[col].dtype != 'object']
for col in binary_categoricals:
    if col in numerical_cols:
        numerical_cols.remove(col)
        categorical_cols.append(col)

# COMMAND ----------

# MAGIC %md
# MAGIC generating synthetic data

# COMMAND ----------

final_synthetic=generate_synthetic_data(df_cleaned, categorical_cols, numerical_cols, 5000, spark, synthetic_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC cleanin_synthetic_data

# COMMAND ----------

synthetic_cleaned=spark.read.format("delta").load(f"{cleaned_data_path}/synthetic_cleaned")

# COMMAND ----------

synthetic_cleaned=clean_synthetic_data(final_synthetic, spark, cleaned_data_path)

# COMMAND ----------

synthetic_cleaned.printSchema()

# COMMAND ----------

synthetic_cleaned=spark.read.format("delta").load(f"{cleaned_data_path}/synthetic_cleaned")
testing=train_pipeline(synthetic_cleaned)


# COMMAND ----------

# MAGIC %md
# MAGIC Listing models

# COMMAND ----------

plot_roc_curve(
    y_true=testing["Logistic Regression"]["y_val"],
    y_proba=testing["Logistic Regression"]["y_proba"]
)

# COMMAND ----------


connection_string = "your_connection_string"

container_name = "your_container_name"

model_df = list_models_in_container(connection_string, container_name)
display(model_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Specific Model loading

# COMMAND ----------

connection_string = "DefaultEndpointsProtocol=https;AccountName=datapreprocesseing;AccountKey=pygFOSK/+wQge0aTj+CPzjmq0o1xfQDdWHJDccZIvSqCT7dFjKBiHcbZybhuWd29y/ZyofmzCQ8O+AStEuJnKA==;EndpointSuffix=core.windows.net"

container_name = "models"
blob_name = "logreg_model_fe0b76eb.pkl"

model = load_model_from_azure(container_name, blob_name, connection_string)

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC Latest Model Loading

# COMMAND ----------


Latest_model = load_latest_model_from_azure(connection_string, container_name)

# COMMAND ----------

print("✅ Model expects these features:\n", model.feature_names_in_)

# COMMAND ----------

result_path = run_inference_pipeline(test_data_path, connection_string, container_name, output_dir)

# COMMAND ----------

