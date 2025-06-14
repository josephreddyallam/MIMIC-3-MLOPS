import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# 👇 Update this path to the root of your repo
sys.path.append("/Workspace/Users/ballam@gitam.in/project_mimic")


from scripts.azure_blob_utils import download_latest_model

sys.path.append("/Workspace/Users/ballam@gitam.in/project_mimic/src/inference")

from inference_pipeline import run_inference_pipeline


# Azure Blob config
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=datapreprocesseing;AccountKey=pygFOSK/+wQge0aTj+CPzjmq0o1xfQDdWHJDccZIvSqCT7dFjKBiHcbZybhuWd29y/ZyofmzCQ8O+AStEuJnKA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "models"
MODEL_PATH = "model.pkl"

@st.cache_resource
def fetch_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📦 Downloading model from Azure Blob Storage...")
        model, model_name = download_latest_model(CONNECTION_STRING, CONTAINER_NAME)
        joblib.dump(model, MODEL_PATH)
        st.success(f"✅ Model downloaded and saved as: {MODEL_PATH}")
    else:
        model = joblib.load(MODEL_PATH)
    return model

# Load model
model = fetch_model()

st.title("🧠 Hospital Mortality Prediction")

st.markdown("Fill in the patient details below to predict mortality risk.")

# Example input fields (you can expand as per model requirements)
age = st.number_input("Age", min_value=0, max_value=120, value=65)
avg_heart_rate = st.number_input("Average Heart Rate", min_value=0, value=80)
avg_dbp = st.number_input("Average Diastolic BP", min_value=0, value=70)
avg_sbp = st.number_input("Average Systolic BP", min_value=0, value=120)
diagnosis = st.number_input("Diagnosis Code (e.g., 90)", min_value=0, value=90)

# Add all other required features with default/fake values or let users select them

# When the user clicks predict
if st.button("Predict"):
    # Construct input
    input_dict = {
        'diagnosis': diagnosis,
        'has_chartevents_data': 1,
        'length_of_stay': 3.2,
        'was_in_ED': 1,
        'ed_duration_hrs': 5.4,
        'age': age,
        'los': 3.2,
        'avg_heart_rate': avg_heart_rate,
        'avg_dbp': avg_dbp,
        'avg_sbp': avg_sbp,
        'avg_oxygen_saturation': 96,
        'avg_resp_rate': 18,
        'avg_creatinine': 1.1,
        'avg_bun': 14,
        'avg_sodium': 138,
        'avg_potassium': 4.0,
        'Diabetes': 1,
        'Hypertension': 0,
        'Heart_Failure': 1,
        'Renal_Failure': 0,
        'admission_type_EMERGENCY': 1,
        'admission_type_URGENT': 0,
        'admission_location_EMERGENCY_ROOM_ADMIT': 1,
        'admission_location_PHYS_REFERRAL/NORMAL_DELI': 0,
        'admission_location_TRANSFER_FROM_HOSP/EXTRAM': 0,
        'admission_location_TRANSFER_FROM_SKILLED_NUR': 0,
        'discharge_location_DISCH-TRAN_TO_PSYCH_HOSP': 0,
        'discharge_location_HOME': 1,
        'discharge_location_HOME_HEALTH_CARE': 0,
        'discharge_location_HOME_WITH_HOME_IV_PROVIDR': 0,
        'discharge_location_HOSPICE-HOME': 0,
        'discharge_location_ICF': 0,
        'discharge_location_LONG_TERM_CARE_HOSPITAL': 0,
        'discharge_location_REHAB/DISTINCT_PART_HOSP': 0,
        'discharge_location_SNF': 0,
        'insurance_Medicaid': 0,
        'insurance_Medicare': 1,
        'insurance_Private': 0,
        'marital_status_MARRIED': 1,
        'marital_status_SEPARATED': 0,
        'marital_status_SINGLE': 0,
        'marital_status_UNKNOWN_(DEFAULT)': 0,
        'marital_status_WIDOWED': 0,
        'ethnicity_ASIAN': 0,
        'ethnicity_BLACK/AFRICAN_AMERICAN': 0,
        'ethnicity_HISPANIC_OR_LATINO': 0,
        'ethnicity_HISPANIC/LATINO_-_PUERTO_RICAN': 0,
        'ethnicity_OTHER': 1,
        'ethnicity_UNKNOWN/NOT_SPECIFIED': 0,
        'ethnicity_WHITE': 0,
        'gender_M': 1
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("❌ High risk: Patient is predicted to **expire during hospitalization**.")
    else:
        st.success("✅ Low risk: Patient is predicted to **survive hospitalization**.")