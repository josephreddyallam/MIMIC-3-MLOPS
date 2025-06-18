# synthetic_generation.py

def generate_synthetic_data(df_cleaned, categorical_cols, numerical_cols, num_rows, spark_session, synthetic_path):
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    # Convert to Pandas
    df = df_cleaned.toPandas()

    # Define SDV metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    for col in categorical_cols:
        metadata.update_column(column_name=col, sdtype='categorical')
    for col in numerical_cols:
        metadata.update_column(column_name=col, sdtype='numerical')

    # Distributions
    numerical_distributions = {col: 'truncnorm' for col in numerical_cols}
    if 'age' in numerical_distributions:
        numerical_distributions['age'] = 'gaussian_kde'
    if 'avg_sodium' in numerical_distributions:
        numerical_distributions['avg_sodium'] = 'gaussian_kde'

    # Synthesize
    synthesizer = GaussianCopulaSynthesizer(metadata, numerical_distributions=numerical_distributions)
    synthesizer.fit(df)
    synthetic_df = synthesizer.sample(num_rows=num_rows)

    # Save to Spark
    synthetic_spark_df = spark_session.createDataFrame(synthetic_df)
    synthetic_spark_df.write.format("delta").mode("overwrite").save(f"{synthetic_path}/final_synthetic")

    return synthetic_spark_df


    # synthetic_cleaning.py

def clean_synthetic_data(df_synthetic, spark_session, clean_path):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PowerTransformer

    # Convert to Pandas
    df = df_synthetic.toPandas()

    # Detect column types
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]

    # Impute missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    for col in categorical_cols:
        df[col].fillna(df[col].mode().iloc[0], inplace=True)

    #  Outlier Capping (IQR Method)
    for col in numerical_cols:
        if col not in binary_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound,
                               np.where(df[col] > upper_bound, upper_bound, df[col]))

    #  Skewness Handling
    for col in numerical_cols:
        if col not in binary_cols:
            skew_val = df[col].skew()
            try:
                if skew_val > 2 or skew_val < -2:
                    df[col] = PowerTransformer(method='yeo-johnson').fit_transform(df[[col]])
                elif skew_val > 1:
                    df[col] = np.log1p(df[col] + 1e-5)
                elif skew_val < -1:
                    df[col] = np.square(df[col])
                elif abs(skew_val) > 0.5:
                    df[col] = PowerTransformer(method='yeo-johnson').fit_transform(df[[col]])
            except Exception:
                continue

    # Save cleaned synthetic data
    cleaned_spark_df = spark_session.createDataFrame(df)
    cleaned_spark_df.write.format("delta").mode("overwrite").save(f"{clean_path}/synthetic_cleaned")

    return cleaned_spark_df
