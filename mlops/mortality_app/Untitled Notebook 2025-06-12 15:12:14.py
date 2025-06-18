# Databricks notebook source
import sys
sys.path.append("/Workspace/Users/ballam@gitam.in/project_mimic/src")

from inference_pipeline import run_inference_pipeline
print("✅ Successfully imported")

# COMMAND ----------

# MAGIC %sh
# MAGIC find /Workspace/Users/ballam@gitam.in/project_mimic -type f -name "*inference*.py"

# COMMAND ----------

