# MIMIC-3-MLOPS
Cloud based Machine Learning model for mortality prediction 


This is my first end-to-end MLOps project focused on building a mortality prediction model for hospitalized patients using the publicly available MIMIC-3 demo dataset.

The goal is to apply real-world data engineering and machine learning practices in a cloud environment.

🧱 Architecture

The project follows the Medallion Architecture (also known as the Bronze-Silver-Gold architecture):
	•	Bronze Layer: Raw data ingestion from 26 MIMIC-3 CSV files.
	•	Silver Layer: Cleaned, joined, and structured data with integrity checks.
	•	Gold Layer: Feature-enriched, model-ready data used for training and inference.

This layered approach ensures modularity, data quality, and reproducibility.


The ML workflow includes:
	1.	Data Preprocessing:
	•	Missing value handling
	•	Feature encoding (one-hot, label encoding)
	•	Outlier treatment
	•	Scaling/skew correction
	2.	Feature Engineering:
	•	Clinical variable aggregation (vitals, labs, etc.)
	•	Length of stay and ED duration
	•	Diagnosis frequency encoding
	•	Comorbidity flags (e.g., Diabetes, Hypertension)
	3.	Model Training:
	•	Logistic Regression and XGBoost with Optuna hyperparameter tuning
	•	Model evaluation with classification metrics and ROC curve
	4.	Inference Pipeline:
	•	Modular pipeline to load model and expected features from Azure Blob Storage
	•	Preprocessing and real-time prediction using the trained model

 ☁️ Cloud Integration
	•	Platform: Azure Databricks (for scalable Spark-based processing)
	•	Storage: Azure Blob Storage (to store models and data)


 Technologies Used

 Python / Pandas / PySpark / XGBoost / Sklearn / Optuna 
 Azure Databricks / Azure Blob Storage / Delta Lake

## 📁 Repository Structure
('''project_mimic/
└── mlops/
├── cicd/                         # (Planned) CI/CD scripts (not implemented yet)
├── mortality_app/               # Frontend app (currently under development)
├── scripts/                     # Shared utility functions
│   ├── init.py
│   ├── azure_blob_utils.py
│   ├── feature_utils.py
│   └── model_utils.py
├── src/                         # Source code for training, preprocessing, inference
│   ├── training/
│   │   ├── run.py
│   │   ├── Setup.py
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
│       └── prediction_accuracy.py''')



if you using this code pass your  accesskey and blob account names in src/pre_processing/Storage_connections.py
