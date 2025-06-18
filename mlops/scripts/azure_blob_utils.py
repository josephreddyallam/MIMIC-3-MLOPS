from azure.storage.blob import BlobServiceClient
import os
import joblib
import tempfile
from typing import Tuple

def download_file_from_blob(connection_string: str, container_name: str, blob_name: str) -> str:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)

    local_path = os.path.join(tempfile.gettempdir(), blob_name)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    return local_path

def download_latest_model(connection_string: str, container_name: str) -> Tuple[object, str]:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    blobs = list(container_client.list_blobs())
    if not blobs:
        raise Exception("No model files found in the container.")

    # Sort by last modified (descending)
    blobs.sort(key=lambda b: b.last_modified, reverse=True)
    latest_blob = blobs[0]

    # Download
    blob_client = container_client.get_blob_client(latest_blob.name)
    local_path = os.path.join(tempfile.gettempdir(), latest_blob.name)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    # Load with joblib
    model = joblib.load(local_path)
    return model, latest_blob.name