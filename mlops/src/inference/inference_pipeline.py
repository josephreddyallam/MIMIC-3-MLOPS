import pandas as pd
import numpy as np
import pickle
import joblib
import os
import tempfile
from typing import Tuple, Optional
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add scripts path
sys.path.append("/Workspace/Users/ballam@gitam.in/project_mimic/mlops/scripts")
from azure_blob_utils import download_latest_model


def load_model(connection_string: str, container_name: str) -> Tuple[object, list]:
    model, model_name = download_latest_model(connection_string, container_name)
    print(f"✅ Loaded model: {model_name}")

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        raise AttributeError("Model does not contain `feature_names_in_`. Ensure it's saved from sklearn >= 1.0")

    return model, expected_features


def preprocess_input(input_df: pd.DataFrame, max_onehot_categories: int = 10) -> pd.DataFrame:
    df = input_df.copy()

    # Drop datetime columns
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    df.drop(columns=datetime_cols, inplace=True)

    # Encode categoricals
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > max_onehot_categories:
            freq_map = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_map)
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    df.columns = df.columns.str.replace(" ", "_")
    return df


def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]


def predict(input_df: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray]:
    processed_df = preprocess_input(input_df)

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        raise AttributeError("Model does not contain `feature_names_in_`. Cannot align features.")

    aligned_df = align_features(processed_df, expected_features)
    probs = model.predict_proba(aligned_df)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
