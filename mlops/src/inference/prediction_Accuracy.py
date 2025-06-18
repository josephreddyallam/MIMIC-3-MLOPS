# Databricks notebook source
# MAGIC %fs ls dbfs:/tmp/

# COMMAND ----------

# Load Delta predictions
df = spark.read.format("delta").load("dbfs:/tmp/inference_test_output")

# Convert to Pandas
df_pd = df.toPandas()



# COMMAND ----------

df_pd.display()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df_pd[["predicted_mortality", "hospital_expire_flag"]].display()

# COMMAND ----------

(df_pd["predicted_mortality"] != df_pd["hospital_expire_flag"]).sum()

# COMMAND ----------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(df_pd["hospital_expire_flag"], df_pd["predicted_mortality"])
print(f"Accuracy: {accuracy:.2f}")

# COMMAND ----------

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(df_pd["hospital_expire_flag"], df_pd["predicted_mortality"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(df_pd["hospital_expire_flag"], df_pd["predicted_mortality"]))

# COMMAND ----------

