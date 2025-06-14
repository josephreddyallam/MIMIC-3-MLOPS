# Databricks notebook source
from azure.storage.blob import BlobServiceClient
import pandas as pd

def list_models_in_container(connection_string: str, container_name: str, prefix: str = "") -> pd.DataFrame:
    # Connect to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # List all blobs in the container
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    
    models = []
    for blob in blob_list:
        models.append({
            "name": blob.name,
            "last_modified": blob.last_modified,
            "size (KB)": round(blob.size / 1024, 2)
        })
    
    # Convert to DataFrame and sort by modified date
    df = pd.DataFrame(models)
    df_sorted = df.sort_values(by="last_modified", ascending=False).reset_index(drop=True)
    
    return df_sorted

# COMMAND ----------

