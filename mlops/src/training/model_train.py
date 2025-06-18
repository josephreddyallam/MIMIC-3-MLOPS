# Databricks notebook source
# MAGIC %run /Workspace/Users/ballam@gitam.in/project_mimic/mlops/src/training/save_models.py

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
import optuna
import joblib
import uuid
import numpy as np

# Function to save and upload model to Azure Blob Storage
def save_and_upload_model(model, model_name, blob_folder_path):
    filename = f"{model_name}_{uuid.uuid4().hex[:8]}.pkl"
    local_path = f"/tmp/{filename}"

    # ✅ Save with joblib
    joblib.dump(model, local_path)

    # Upload to Azure
    blob_path = blob_folder_path + filename
    dbutils.fs.cp(f"file:{local_path}", blob_path)
    return blob_path

# Feature selection helpers
def drop_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def select_top_features_with_selectkbest(X, y, k=30):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return pd.DataFrame(X_new, columns=selected_features), selected_features

def select_features_with_rfe(X, y, n_features=20):
    model = LogisticRegression(solver='liblinear')
    selector = RFE(model, n_features_to_select=n_features)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()
    return X[selected_features], selected_features

# Function to encode, handle categoricals, and split data
def encode_and_split(df_spark, target_col='hospital_expire_flag', max_onehot_categories=10):
    df = df_spark.toPandas()

    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
    df = df.drop(columns=datetime_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        n_unique = X[col].nunique()
        if n_unique > max_onehot_categories:
            freq_map = X[col].value_counts().to_dict()
            X[col] = X[col].map(freq_map)
        else:
            X = pd.get_dummies(X, columns=[col], drop_first=True)

    X.columns = X.columns.str.replace(" ", "_")

    # Step 1: Drop highly correlated features
    X = drop_highly_correlated_features(X, threshold=0.9)

    # Step 2: Split initially for feature selection
    X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 3: Select features using SelectKBest
    X_train_kbest, kbest_features = select_top_features_with_selectkbest(X_train_full, y_train, k=50)
    X_val_kbest = X_val_full[kbest_features]

    # Step 4: Further reduce using RFE
    X_train, rfe_features = select_features_with_rfe(X_train_kbest, y_train, n_features=30)
    X_val = X_val_kbest[rfe_features]

    return X_train, X_val, y_train, y_val

# Model evaluation
def evaluate_model(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }

# Optuna for Logistic Regression
def tune_logistic_regression(X_train, y_train, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            "C": trial.suggest_loguniform("C", 1e-4, 1e2),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"])
        }
        try:
            model = LogisticRegression(**params, max_iter=1000)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_proba)
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# 🔥 Final Training Pipeline
def train_pipeline(df_spark, target_column='hospital_expire_flag'):
    models_path = "wasbs://models@datapreprocesseing.blob.core.windows.net/"

    # Step: Encode & Split
    X_train, X_val, y_train, y_val = encode_and_split(df_spark, target_col=target_column)

    # 🔍 Tune Logistic Regression
    print("🔍 Tuning Logistic Regression with Optuna...")
    best_lr_params = tune_logistic_regression(X_train, y_train, X_val, y_val)
    lr_model = LogisticRegression(**best_lr_params, max_iter=1000)
    lr_model.fit(X_train, y_train)

    # 📈 Predict & Evaluate
    y_pred_lr = lr_model.predict(X_val)
    y_proba_lr = lr_model.predict_proba(X_val)[:, 1]
    lr_results = evaluate_model(y_val, y_pred_lr, y_proba_lr)

    # 📋 Classification Report
    print("\n📋 Classification Report: Logistic Regression")
    print(classification_report(y_val, y_pred_lr))

    # 🔳 Confusion Matrix
    sns.heatmap(confusion_matrix(y_val, y_pred_lr), annot=True, fmt='d', cmap='Greens')
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 💾 Save model to Azure
    lr_blob_path = save_and_upload_model(lr_model, "logreg_model", models_path)

    # ✅ Return only Logistic Regression info
    return {
        "Logistic Regression": {
            "model": lr_model,
            "params": best_lr_params,
            "metrics": lr_results,
            "blob_path": lr_blob_path,
            "X_val": X_val,
            "y_val": y_val,
            "y_proba": y_proba_lr
        }
    }

# COMMAND ----------

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_proba, label="Logistic Regression"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc_score(y_true, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

# COMMAND ----------

