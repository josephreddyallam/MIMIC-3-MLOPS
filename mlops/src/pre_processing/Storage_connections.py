# Databricks notebook source
# Azure Blob Storage configuration
blob_account = "datapreprocesseing"
access_key = "pygFOSK/+wQge0aTj+CPzjmq0o1xfQDdWHJDccZIvSqCT7dFjKBiHcbZybhuWd29y/ZyofmzCQ8O+AStEuJnKA=="

# Set Spark config to authenticate with Azure Blob Storage
spark.conf.set(
    f"fs.azure.account.key.{blob_account}.blob.core.windows.net",
    access_key
)


containers = [
    "rawdata",
    "processeddata",
    "syntheticdata",
    "cleaneddata",
    "models",         
    "reports",
    "predictions",
    "logs",
    "predictions"
]

# Generate container paths dictionary
blob_paths = {
    container: f"wasbs://{container}@{blob_account}.blob.core.windows.net/"
    for container in containers
}

# Assign container paths to variables
raw_data_path = blob_paths["rawdata"]
processed_data_path = blob_paths["processeddata"]
synthetic_data_path = blob_paths["syntheticdata"]
cleaned_data_path = blob_paths["cleaneddata"]
models_path = blob_paths["models"]           
reports_path = blob_paths["reports"]
predictions_path = blob_paths["predictions"]
logs_path = blob_paths["logs"]
predictions_path = blob_paths["predictions"]

# Optional: Print all container paths for confirmation
for name, path in blob_paths.items():
    print(f"{name}: {path}")

# COMMAND ----------

