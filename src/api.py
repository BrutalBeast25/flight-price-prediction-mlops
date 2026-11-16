
import os
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any


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

MODEL_VERSION = "1.0"


# ---------------------------------------------------------
# 2. LOAD MODEL AND FEATURE SCHEMA
# ---------------------------------------------------------

print("\nLoading Flight Price Prediction model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model artifact not found: {}".format(MODEL_PATH)
    )

if not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError(
        "Feature schema not found: {}".format(SCHEMA_PATH)
    )

pipeline = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as file:
    feature_schema = json.load(file)

EXPECTED_FEATURES = [
    feature["name"]
    for feature in feature_schema["features"]
]

FEATURE_DTYPES = {
    feature["name"]: feature["dtype"]
    for feature in feature_schema["features"]
}

print("Model loaded successfully.")
print("Expected features: {}".format(len(EXPECTED_FEATURES)))


# ---------------------------------------------------------
# 3. INITIALIZE FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Flight Price Prediction API",
    description=(
        "REST API for predicting flight ticket prices "
        "using an optimized XGBoost model."
    ),
    version=MODEL_VERSION
)


# ---------------------------------------------------------
# 4. DEFINE REQUEST SCHEMA
# ---------------------------------------------------------

class PredictionRequest(BaseModel):

    features: Dict[str, Any]


# ---------------------------------------------------------
# 5. HEALTH CHECK ENDPOINT
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Optimized XGBoost",
        "model_version": MODEL_VERSION
    }


# ---------------------------------------------------------
# 6. INPUT VALIDATION
# ---------------------------------------------------------

def validate_input(features):

    received_features = set(features.keys())

    expected_features = set(EXPECTED_FEATURES)

    missing_features = (
        expected_features - received_features
    )

    unexpected_features = (
        received_features - expected_features
    )

    if missing_features or unexpected_features:

        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input feature mismatch.",
                "missing_features": sorted(missing_features),
                "unexpected_features": sorted(unexpected_features)
            }
        )

    validated_features = {}

    for feature_name in EXPECTED_FEATURES:

        value = features[feature_name]
        expected_dtype = FEATURE_DTYPES[feature_name]

        if value is None:

            raise HTTPException(
                status_code=422,
                detail="Feature '{}' cannot be null.".format(
                    feature_name
                )
            )

        try:

            if expected_dtype.startswith("int"):

                numeric_value = float(value)

                if not np.isfinite(numeric_value):
                    raise ValueError("Non-finite value")

                if not numeric_value.is_integer():
                    raise ValueError("Expected integer value")

                validated_features[feature_name] = int(
                    numeric_value
                )

            elif expected_dtype.startswith("float"):

                numeric_value = float(value)

                if not np.isfinite(numeric_value):
                    raise ValueError("Non-finite value")

                validated_features[feature_name] = numeric_value

            elif expected_dtype == "bool":

                if not isinstance(value, bool):
                    raise ValueError("Expected boolean value")

                validated_features[feature_name] = value

            else:

                validated_features[feature_name] = value

        except (ValueError, TypeError, OverflowError):

            raise HTTPException(
                status_code=422,
                detail=(
                    "Invalid value for feature '{}'. "
                    "Expected type: {}."
                ).format(feature_name, expected_dtype)
            )

    return validated_features


# ---------------------------------------------------------
# 7. PREDICTION ENDPOINT
# ---------------------------------------------------------

@app.post("/predict")
def predict_price(request: PredictionRequest):

    try:

        validated_features = validate_input(
            request.features
        )

        input_data = pd.DataFrame(
            [validated_features],
            columns=EXPECTED_FEATURES
        )

        prediction = pipeline.predict(input_data)

        predicted_price = float(prediction[0])

        if not np.isfinite(predicted_price):
            raise ValueError(
                "Model generated an invalid prediction."
            )

        return {
            "status": "success",
            "model_version": MODEL_VERSION,
            "predicted_price": round(predicted_price, 2)
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed: {}".format(str(error))
        )
