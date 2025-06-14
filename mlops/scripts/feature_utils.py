import pandas as pd

def preprocess_for_inference(df: pd.DataFrame):
    """
    Apply same preprocessing steps used during training.
    """
    # Example placeholder:
    X = df.drop(columns=["target"], errors="ignore")  # Modify as per your schema
    y = df["target"] if "target" in df.columns else None
    return X, y