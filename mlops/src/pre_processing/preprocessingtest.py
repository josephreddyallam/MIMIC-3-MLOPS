# Databricks notebook source
def preprocess_and_store_raw(df_raw, processed_path, spark_session):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PowerTransformer

    df = df_raw.toPandas()
    df.drop(columns=["icustay_id", "subject_id", "hadm_id"], inplace=True)

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']

    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_categoricals = ['was_in_ED', 'Diabetes', 'Hypertension', 'Heart_Failure', 'Renal_Failure', 'has_chartevents_data']

    for col in binary_categoricals:
        if col in numerical_cols:
            numerical_cols.remove(col)
            categorical_cols.append(col)

    for col in numerical_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.where(X[col] < lower_bound, lower_bound,
                          np.where(X[col] > upper_bound, upper_bound, X[col]))

    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    for col in categorical_cols:
        X[col].fillna(X[col].mode().iloc[0], inplace=True)

    if 'age' in X.columns:
        X['age'] = X['age'] / 365.25

    binary_cols = [col for col in X.columns if X[col].nunique() == 2]
    true_numerical_cols = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and col not in binary_cols]

    for col in true_numerical_cols:
        skew_val = X[col].skew()
        try:
            if skew_val > 2 or skew_val < -2:
                X[col] = PowerTransformer(method='yeo-johnson').fit_transform(X[[col]])
            elif skew_val > 1:
                X[col] = np.log1p(X[col] + 1e-5)
            elif skew_val < -1:
                X[col] = np.square(X[col])
            elif abs(skew_val) > 0.5:
                X[col] = PowerTransformer(method='yeo-johnson').fit_transform(X[[col]])
        except Exception:
            pass

    df[X.columns] = X
    cleaned_spark_df = spark_session.createDataFrame(df)
    cleaned_spark_df.write.format("delta").mode("overwrite").save(f"{processed_path}/df_cleaned")

    return cleaned_spark_df

# COMMAND ----------

