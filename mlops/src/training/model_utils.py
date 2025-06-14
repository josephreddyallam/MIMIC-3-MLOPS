# Databricks notebook source
from azure.storage.blob import BlobClient
import pickle

def load_model_from_azure(container_name, blob_name, connection_string):
    blob_client = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name
    )

    downloader = blob_client.download_blob()
    model_data = downloader.readall()
    model = pickle.loads(model_data)
    return model

# COMMAND ----------

