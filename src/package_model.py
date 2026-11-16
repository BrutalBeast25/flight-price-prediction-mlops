
import os
import json
import platform

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
import xgboost

from mlflow.tracking import MlflowClient


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "v1.0"
)

MLFLOW_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

MLFLOW_TRACKING_URI = "sqlite:///{}".format(
    MLFLOW_DB_PATH.replace("\\", "/")
)

EXPERIMENT_NAME = "Flight Price Prediction"
MODEL_ARTIFACT_PATH = "final_xgboost_pipeline"
TARGET_COLUMN = "price"


# ---------------------------------------------------------
# 2. CONFIGURE MLFLOW
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError("MLflow experiment not found.")

print("\nSearching for the final evaluation run...")


# ---------------------------------------------------------
# 3. IDENTIFY THE EXISTING FINAL MODEL RUN
# ---------------------------------------------------------

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        "tags.model_stage = 'final_test_evaluation'"
    ),
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise ValueError(
        "No final test evaluation run was found in MLflow."
    )

run = runs[0]
run_id = run.info.run_id

print("Selected MLflow Run ID: {}".format(run_id))


# ---------------------------------------------------------
# 4. LOAD THE EXISTING MODEL FROM MLFLOW
# ---------------------------------------------------------

model_uri = "runs:/{}/{}".format(
    run_id,
    MODEL_ARTIFACT_PATH
)

print("\nLoading trained model from MLflow...")

pipeline = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully.")


# ---------------------------------------------------------
# 5. PREPARE ARTIFACT DIRECTORY
# ---------------------------------------------------------

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# 6. EXPORT TRAINED MODEL
# ---------------------------------------------------------

model_path = os.path.join(
    ARTIFACT_DIR,
    "model.joblib"
)

joblib.dump(
    pipeline,
    model_path,
    compress=3
)

print("\nModel exported to:")
print(model_path)


# ---------------------------------------------------------
# 7. EXPORT MODEL CONFIGURATION
# ---------------------------------------------------------

model_config = {
    "model_name": "Optimized XGBoost",
    "model_version": "1.0",
    "mlflow_run_id": run_id,
    "optuna_best_trial": run.data.params.get(
        "optuna_best_trial"
    ),
    "optuna_best_cv_rmse": run.data.params.get(
        "optuna_best_cv_rmse"
    ),
    "hyperparameters": {}
}

for key, value in run.data.params.items():

    if key not in [
        "selected_model",
        "optuna_best_trial",
        "optuna_best_cv_rmse",
        "training_records",
        "validation_records",
        "test_records",
        "test_evaluation_only"
    ]:

        try:
            model_config["hyperparameters"][key] = json.loads(
                value
            )
        except (ValueError, TypeError):
            model_config["hyperparameters"][key] = value


config_path = os.path.join(
    ARTIFACT_DIR,
    "model_config.json"
)

with open(config_path, "w") as file:
    json.dump(
        model_config,
        file,
        indent=4
    )

print("Model configuration exported.")


# ---------------------------------------------------------
# 8. EXPORT FEATURE SCHEMA
# ---------------------------------------------------------

train_data = pd.read_csv(TRAIN_DATA_PATH)

X = train_data.drop(
    columns=[TARGET_COLUMN]
)

feature_schema = {
    "target_column": TARGET_COLUMN,
    "feature_count": len(X.columns),
    "features": []
}

for column in X.columns:

    feature_schema["features"].append({
        "name": column,
        "dtype": str(X[column].dtype)
    })


# Verify that the loaded model expects the same features.
if hasattr(pipeline, "feature_names_in_"):

    expected_features = list(
        pipeline.feature_names_in_
    )

    if expected_features != list(X.columns):

        raise ValueError(
            "Model feature names do not match the training schema."
        )


schema_path = os.path.join(
    ARTIFACT_DIR,
    "feature_schema.json"
)

with open(schema_path, "w") as file:
    json.dump(
        feature_schema,
        file,
        indent=4
    )

print("Feature schema exported.")


# ---------------------------------------------------------
# 9. EXPORT EVALUATION METRICS
# ---------------------------------------------------------

evaluation_metrics = {
    "mlflow_run_id": run_id,
    "metrics": run.data.metrics
}

metrics_path = os.path.join(
    ARTIFACT_DIR,
    "evaluation_metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(
        evaluation_metrics,
        file,
        indent=4
    )

print("Evaluation metrics exported.")


# ---------------------------------------------------------
# 10. EXPORT ENVIRONMENT INFORMATION
# ---------------------------------------------------------

environment = {
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "mlflow": mlflow.__version__,
    "joblib": joblib.__version__
}

environment_path = os.path.join(
    ARTIFACT_DIR,
    "environment.json"
)

with open(environment_path, "w") as file:
    json.dump(
        environment,
        file,
        indent=4
    )


# ---------------------------------------------------------
# 11. COMPLETION
# ---------------------------------------------------------

print("\n========================================")
print("MODEL PACKAGING COMPLETED")
print("========================================")

print("\nArtifact directory:")
print(ARTIFACT_DIR)

print("\nGenerated files:")

for filename in os.listdir(ARTIFACT_DIR):
    print("- {}".format(filename))
