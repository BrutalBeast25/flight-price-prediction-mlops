
import os
import warnings

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import create_pipeline


warnings.filterwarnings("ignore", category=UserWarning)


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

TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test_features.csv"
)

MLFLOW_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

OPTUNA_XGB_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "optuna_xgboost.db"
)

MLFLOW_TRACKING_URI = "sqlite:///{}".format(
    MLFLOW_DB_PATH.replace("\\", "/")
)

OPTUNA_XGB_STORAGE = "sqlite:///{}".format(
    OPTUNA_XGB_DB_PATH.replace("\\", "/")
)

EXPERIMENT_NAME = "Flight Price Prediction"
STUDY_NAME = "xgboost_optimization"

TARGET_COLUMN = "price"
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. LOAD TRAINING DATA
# ---------------------------------------------------------

print("\nLoading training data...")

train_data = pd.read_csv(TRAIN_DATA_PATH)

X = train_data.drop(columns=[TARGET_COLUMN])
y = train_data[TARGET_COLUMN]

print("Total training records : {}".format(len(X)))


# ---------------------------------------------------------
# 3. RECREATE THE ORIGINAL TRAIN/VALIDATION SPLIT
# ---------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=X["class"]
)

print("\nTraining records   : {}".format(len(X_train)))
print("Validation records : {}".format(len(X_val)))


# ---------------------------------------------------------
# 4. LOAD UNTOUCHED TEST DATA
# ---------------------------------------------------------

print("\nLoading final test dataset...")

test_data = pd.read_csv(TEST_DATA_PATH)

if TARGET_COLUMN not in test_data.columns:

    raise ValueError(
        "Target column '{}' is missing from the test dataset.".format(
            TARGET_COLUMN
        )
    )

X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]

if list(X_train.columns) != list(X_test.columns):

    raise ValueError(
        "Training and test feature columns do not match."
    )

print("Test records       : {}".format(len(X_test)))


# ---------------------------------------------------------
# 5. LOAD BEST XGBOOST PARAMETERS FROM OPTUNA
# ---------------------------------------------------------

print("\nLoading best XGBoost parameters from Optuna...")

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=OPTUNA_XGB_STORAGE
)

best_params = study.best_params

print("Best Trial Number : {}".format(
    study.best_trial.number
))

print("Best CV RMSE      : {:.4f}".format(
    study.best_value
))

print("\nBest Hyperparameters:")

for key, value in best_params.items():

    print("{} : {}".format(key, value))


# ---------------------------------------------------------
# 6. CONFIGURE MLFLOW
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# 7. BUILD AND TRAIN SELECTED XGBOOST MODEL
# ---------------------------------------------------------

print("\n----------------------------------------")
print("TRAINING SELECTED XGBOOST MODEL")
print("----------------------------------------")

xgb_model = XGBRegressor(
    **best_params,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = create_pipeline(
    model=xgb_model,
    scale_numeric=False
)

pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed.")


# ---------------------------------------------------------
# 8. FINAL TEST DATASET PREDICTIONS
# ---------------------------------------------------------

print("\nGenerating predictions on final test dataset...")

test_predictions = pipeline.predict(X_test)


# ---------------------------------------------------------
# 9. CALCULATE FINAL TEST METRICS
# ---------------------------------------------------------

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ---------------------------------------------------------
# 10. ALSO CALCULATE TRAINING AND VALIDATION METRICS
# ---------------------------------------------------------

train_predictions = pipeline.predict(X_train)

val_predictions = pipeline.predict(X_val)

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_rmse = mean_squared_error(
    y_train,
    train_predictions
) ** 0.5

train_r2 = r2_score(
    y_train,
    train_predictions
)

val_mae = mean_absolute_error(
    y_val,
    val_predictions
)

val_rmse = mean_squared_error(
    y_val,
    val_predictions
) ** 0.5

val_r2 = r2_score(
    y_val,
    val_predictions
)


# ---------------------------------------------------------
# 11. DISPLAY FINAL RESULTS
# ---------------------------------------------------------

print("\n========================================")
print("FINAL XGBOOST MODEL PERFORMANCE")
print("========================================")

print("\nTraining Metrics")

print("MAE  : {:.4f}".format(train_mae))
print("RMSE : {:.4f}".format(train_rmse))
print("R2   : {:.6f}".format(train_r2))

print("\nValidation Metrics")

print("MAE  : {:.4f}".format(val_mae))
print("RMSE : {:.4f}".format(val_rmse))
print("R2   : {:.6f}".format(val_r2))

print("\nFinal Test Metrics")

print("MAE  : {:.4f}".format(test_mae))
print("RMSE : {:.4f}".format(test_rmse))
print("R2   : {:.6f}".format(test_r2))


# ---------------------------------------------------------
# 12. LOG FINAL MODEL AND METRICS TO MLFLOW
# ---------------------------------------------------------

print("\nLogging final evaluation to MLflow...")

with mlflow.start_run(
    run_name="Final XGBoost Test Evaluation"
):

    mlflow.log_param(
        "selected_model",
        "Optimized XGBoost"
    )

    mlflow.log_param(
        "optuna_best_trial",
        study.best_trial.number
    )

    mlflow.log_param(
        "optuna_best_cv_rmse",
        study.best_value
    )

    mlflow.log_param(
        "training_records",
        len(X_train)
    )

    mlflow.log_param(
        "validation_records",
        len(X_val)
    )

    mlflow.log_param(
        "test_records",
        len(X_test)
    )

    mlflow.log_param(
        "test_evaluation_only",
        True
    )

    mlflow.log_params(best_params)

    # Training metrics
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_r2", train_r2)

    # Validation metrics
    mlflow.log_metric("val_mae", val_mae)
    mlflow.log_metric("val_rmse", val_rmse)
    mlflow.log_metric("val_r2", val_r2)

    # Final test metrics
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_r2", test_r2)

    mlflow.set_tag(
        "model_stage",
        "final_test_evaluation"
    )

    mlflow.set_tag(
        "selected_model",
        "Optimized XGBoost"
    )

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="final_xgboost_pipeline",
        serialization_format="cloudpickle"
    )

print("\n========================================")
print("FINAL TEST EVALUATION COMPLETED")
print("========================================")

print("\nFinal model and evaluation metrics logged to MLflow.")
