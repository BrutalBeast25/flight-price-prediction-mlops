
# import os
# import json
# import math
# import pytest

# from fastapi.testclient import TestClient
# from src.api import app


# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SAMPLE_INPUT_PATH = os.path.join(
#     PROJECT_ROOT,
#     "artifacts",
#     "v1.0",
#     "sample_input.json"
# )

# with open(SAMPLE_INPUT_PATH, "r") as file:
#     SAMPLE_FEATURES = json.load(file)[0]


# client = TestClient(app)


import os
import sys
import json
import math
import pytest

# Add project root to Python import path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from src.api import app

SAMPLE_INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "v1.0",
    "sample_input.json"
)

with open(SAMPLE_INPUT_PATH, "r") as file:
    SAMPLE_FEATURES = json.load(file)[0]

client = TestClient(app)



# ---------------------------------------------------------
# 1. HEALTH CHECK TEST
# ---------------------------------------------------------

def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "Optimized XGBoost"
    assert data["model_version"] == "1.0"


# ---------------------------------------------------------
# 2. VALID PREDICTION TEST
# ---------------------------------------------------------

def test_valid_prediction():

    response = client.post(
        "/predict",
        json={"features": SAMPLE_FEATURES}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["model_version"] == "1.0"

    assert isinstance(data["predicted_price"], (int, float))
    assert math.isfinite(data["predicted_price"])
    assert data["predicted_price"] > 0


# ---------------------------------------------------------
# 3. MISSING FEATURE TEST
# ---------------------------------------------------------

def test_missing_feature():

    features = SAMPLE_FEATURES.copy()

    features.pop("airline")

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert "airline" in detail["missing_features"]


# ---------------------------------------------------------
# 4. UNEXPECTED FEATURE TEST
# ---------------------------------------------------------

def test_unexpected_feature():

    features = SAMPLE_FEATURES.copy()

    features["unexpected_feature"] = "test"

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert "unexpected_feature" in detail["unexpected_features"]


# ---------------------------------------------------------
# 5. NULL VALUE TEST
# ---------------------------------------------------------

def test_null_feature():

    features = SAMPLE_FEATURES.copy()

    features["airline"] = None

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422

    assert "cannot be null" in response.json()["detail"]


# ---------------------------------------------------------
# 6. INVALID NUMERIC VALUE TEST
# ---------------------------------------------------------

def test_invalid_numeric_value():

    features = SAMPLE_FEATURES.copy()

    features["days_left"] = "invalid"

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422
