# Databricks notebook source
import os
import pickle
import uuid
from azure.storage.blob import BlobServiceClient

def save_and_upload_model_to_azure(model, model_name, container_name, connection_string):
    # Generate unique filename
    filename = f"{model_name}_{uuid.uuid4().hex[:8]}.pkl"
    local_path = f"/tmp/{filename}"

    # Save model to local file
    with open(local_path, "wb") as f:
        pickle.dump(model, f)

    # Connect to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it doesn't exist
    try:
        container_client.create_container()
    except Exception:
        pass  # Already exists

    # Upload model to Azure Blob
    blob_client = container_client.get_blob_client(blob=filename)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    # Return blob URL
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{filename}"
    return blob_url

# COMMAND ----------

