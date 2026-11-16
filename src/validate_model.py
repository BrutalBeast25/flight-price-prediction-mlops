
import os
import json

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "v1.0"
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "model.joblib"
)

SCHEMA_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_schema.json"
)

TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

SAMPLE_INPUT_PATH = os.path.join(
    ARTIFACT_DIR,
    "sample_input.json"
)


# ---------------------------------------------------------
# 2. LOAD FEATURE SCHEMA
# ---------------------------------------------------------

print("\nLoading feature schema...")

with open(SCHEMA_PATH, "r") as file:
    feature_schema = json.load(file)

expected_features = [
    feature["name"]
    for feature in feature_schema["features"]
]

print("Expected feature count: {}".format(
    len(expected_features)
))


# ---------------------------------------------------------
# 3. PREPARE A REPRESENTATIVE INPUT
# ---------------------------------------------------------

print("\nPreparing sample input...")

train_data = pd.read_csv(TRAIN_DATA_PATH)

X = train_data.drop(
    columns=[feature_schema["target_column"]]
)

if list(X.columns) != expected_features:
    raise ValueError(
        "Training features do not match the saved feature schema."
    )

sample_input = X.iloc[[0]].copy()

# Save the sample so it can be reused independently.
sample_input.to_json(
    SAMPLE_INPUT_PATH,
    orient="records",
    indent=4
)

print("Sample input saved successfully.")


# ---------------------------------------------------------
# 4. LOAD THE PACKAGED MODEL
# ---------------------------------------------------------

print("\nLoading local model artifact...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# 5. VALIDATE INPUT FEATURES
# ---------------------------------------------------------

print("\nValidating input features...")

with open(SAMPLE_INPUT_PATH, "r") as file:
    sample_records = json.load(file)

if len(sample_records) != 1:
    raise ValueError(
        "Expected exactly one sample input record."
    )

input_data = pd.DataFrame(sample_records)

if list(input_data.columns) != expected_features:
    raise ValueError(
        "Input features do not match the expected schema."
    )

print("Feature validation successful.")


# ---------------------------------------------------------
# 6. GENERATE PREDICTION
# ---------------------------------------------------------

print("\nGenerating prediction...")

prediction = pipeline.predict(input_data)

if not np.isfinite(prediction).all():
    raise ValueError(
        "Model returned an invalid numerical prediction."
    )

predicted_price = float(prediction[0])

print("\n========================================")
print("MODEL INFERENCE VALIDATION")
print("========================================")

print("Model version       : 1.0")
print("Input features      : {}".format(len(input_data.columns)))
print("Prediction          : {:.2f}".format(predicted_price))

print("\n========================================")
print("MODEL VALIDATION SUCCESSFUL")
print("========================================")
