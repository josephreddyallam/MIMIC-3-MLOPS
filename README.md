# 🏥 MIMIC-3-MLOPS: Cloud-Based Mortality Prediction

This project is my first end-to-end MLOps solution focused on predicting **hospital mortality** using the publicly available **MIMIC-III demo dataset**. It is built entirely in a cloud environment using **Azure Databricks** and follows modern data engineering and ML best practices.

---

## 📌 Project Objective

Develop a scalable, modular, and production-ready machine learning pipeline to predict mortality risk in ICU patients using real clinical data.

---

## 🧱 Architecture: Medallion Pattern

The project is structured using the **Medallion Architecture**:

- **🥉 Bronze Layer**: Raw ingestion of all 26 MIMIC-III CSV files.
- **🥈 Silver Layer**: Cleaned, validated, and joined datasets.
- **🥇 Gold Layer**: Feature-enriched, model-ready data used for training and inference.

This layered approach ensures:
- Data quality
- Reproducibility
- Maintainability

---

## 🔄 Machine Learning Workflow

### 1️⃣ Data Preprocessing
- Missing value handling
- One-hot and label encoding
- Outlier capping
- Scaling and skew correction

### 2️⃣ Feature Engineering
- Aggregation of vitals and labs
- Length of stay and ED duration features
- Frequency encoding of diagnosis codes
- Comorbidity flags (e.g., Diabetes, Hypertension)

### 3️⃣ Model Training
- Algorithms: **Logistic Regression** and **XGBoost**
- Hyperparameter tuning via **Optuna**
- Evaluation metrics: Accuracy, AUC-ROC, Classification Report

### 4️⃣ Inference Pipeline
- Load trained model and expected features from **Azure Blob Storage**
- Apply preprocessing to new patient data
- Generate real-time mortality predictions

---

## ☁️ Cloud Integration

- **Platform**: Azure Databricks (PySpark and Python)
- **Storage**: Azure Blob Storage
- **Data Format**: Delta Lake

---

## 🧰 Technologies Used

- **Languages**: Python, PySpark
- **Libraries**: pandas, scikit-learn, XGBoost, Optuna
- **Cloud Tools**: Azure Blob Storage, Azure Databricks, Delta Lake

---

## 📁 Repository Structure
##project_mimic/
└── mlops/
├── cicd/                       # (Planned) CI/CD scripts
├── mortality_app/              # Streamlit-based frontend (under development)
├── scripts/                    # Shared utility functions
│   ├── init.py
│   ├── azure_blob_utils.py
│   ├── feature_utils.py
│   └── model_utils.py
├── src/
│   ├── training/
│   │   ├── run.py
│   │   ├── setup.py
│   │   ├── save_models.py
│   │   ├── model_utils.py
│   │   ├── model_train.py
│   │   ├── load_latest_model.py
│   │   └── list_models_in_container.py
│   ├── pre_processing/
│   │   ├── synthetic_data_generation.py
│   │   ├── Storage_connections.py
│   │   └── preprocessing.py
│   └── inference/
│       ├── init.py
│       ├── inference_run.py
│       ├── inference_pipeline.py
│       └── prediction_accuracy.py##
