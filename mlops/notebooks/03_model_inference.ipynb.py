# Databricks notebook source
import sys
sys.path.append("/Workspace/Repos/your-name/ml_project/scripts")
sys.path.append("/Workspace/Repos/your-name/ml_project/src/inference")

from inference_pipeline import run_inference_pipeline

# Parameters
connection_string = "DefaultEndpointsProtocol=https;AccountName=datapreprocesseing;AccountKey=pygFOSK/+wQge0aTj+CPzjmq0o1xfQDdWHJDccZIvSqCT7dFjKBiHcbZybhuWd29y/ZyofmzCQ8O+AStEuJnKA==;EndpointSuffix=core.windows.net"
container_name = "models"
test_data_path = "/dbfs/FileStore/test_data.csv"
output_dir = "/dbfs/FileStore/inference_results"

# Run
metrics = run_inference_pipeline(test_data_path, connection_string, container_name, output_dir)
print("✅ Metrics:", metrics)

# COMMAND ----------

