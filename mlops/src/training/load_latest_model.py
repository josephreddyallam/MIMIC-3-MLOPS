# Databricks notebook source
from azure.storage.blob import BlobServiceClient
import os
import tempfile
import joblib  # or use pickle depending on how the model was saved

def load_latest_model_from_azure(connection_string: str, container_name: str, local_dir: str = "/tmp") -> object:
    """
    Loads the latest model (by last_modified timestamp) from an Azure Blob container.
    Downloads it locally and returns the loaded model.
    """
    # Create blob service and container client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # List and sort blobs by last_modified descending
    blobs = list(container_client.list_blobs())
    if not blobs:
        raise Exception("No models found in the container.")

    blobs.sort(key=lambda b: b.last_modified, reverse=True)
    latest_blob = blobs[0]
    blob_client = container_client.get_blob_client(latest_blob.name)

    # Create local path for the model
    local_model_path = os.path.join(local_dir, latest_blob.name.split('/')[-1])

    # Download the latest model
    with open(local_model_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    # Load and return the model
    model = joblib.load(local_model_path)
    print(f"Loaded model: {latest_blob.name}")
    return model