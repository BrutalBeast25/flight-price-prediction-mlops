
import os
import warnings

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from src.preprocessing import create_pipeline


warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

MLFLOW_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

OPTUNA_RF_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "optuna_random_forest.db"
)

OPTUNA_XGB_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "optuna_xgboost.db"
)

MLFLOW_TRACKING_URI = "sqlite:///{}".format(
    MLFLOW_DB_PATH.replace("\\", "/")
)

OPTUNA_RF_STORAGE = "sqlite:///{}".format(
    OPTUNA_RF_DB_PATH.replace("\\", "/")
)

OPTUNA_XGB_STORAGE = "sqlite:///{}".format(
    OPTUNA_XGB_DB_PATH.replace("\\", "/")
)

EXPERIMENT_NAME = "Flight Price Prediction"

TARGET_COLUMN = "price"
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. LOAD TRAINING DATA
# ---------------------------------------------------------

print("\nLoading processed training data...")

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=X["class"]
)

print("Training records   : {}".format(len(X_train)))
print("Validation records : {}".format(len(X_val)))


# ---------------------------------------------------------
# 3. LOAD BEST OPTUNA PARAMETERS
# ---------------------------------------------------------

print("\nLoading optimized Random Forest parameters...")

rf_study = optuna.load_study(
    study_name="random_forest_optimization",
    storage=OPTUNA_RF_STORAGE
)

rf_best_params = rf_study.best_params

print("Best RF Trial      : {}".format(
    rf_study.best_trial.number
))

print("Best RF CV RMSE    : {:.4f}".format(
    rf_study.best_value
))

print("\nLoading optimized XGBoost parameters...")

xgb_study = optuna.load_study(
    study_name="xgboost_optimization",
    storage=OPTUNA_XGB_STORAGE
)

xgb_best_params = xgb_study.best_params

print("Best XGB Trial     : {}".format(
    xgb_study.best_trial.number
))

print("Best XGB CV RMSE   : {:.4f}".format(
    xgb_study.best_value
))


# ---------------------------------------------------------
# 4. CONFIGURE MLFLOW
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# 5. DEFINE MODEL EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model(model_name, pipeline, X_train, y_train, X_val, y_val):

    print("\n----------------------------------------")
    print("TRAINING {}".format(model_name.upper()))
    print("----------------------------------------")

    pipeline.fit(
        X_train,
        y_train
    )

    train_predictions = pipeline.predict(X_train)

    val_predictions = pipeline.predict(X_val)

    metrics = {
        "train_mae": mean_absolute_error(
            y_train,
            train_predictions
        ),

        "train_rmse": mean_squared_error(
            y_train,
            train_predictions
        ) ** 0.5,

        "train_r2": r2_score(
            y_train,
            train_predictions
        ),

        "val_mae": mean_absolute_error(
            y_val,
            val_predictions
        ),

        "val_rmse": mean_squared_error(
            y_val,
            val_predictions
        ) ** 0.5,

        "val_r2": r2_score(
            y_val,
            val_predictions
        )
    }

    print("\nTraining Metrics")

    print("MAE  : {:.4f}".format(metrics["train_mae"]))
    print("RMSE : {:.4f}".format(metrics["train_rmse"]))
    print("R2   : {:.6f}".format(metrics["train_r2"]))

    print("\nValidation Metrics")

    print("MAE  : {:.4f}".format(metrics["val_mae"]))
    print("RMSE : {:.4f}".format(metrics["val_rmse"]))
    print("R2   : {:.6f}".format(metrics["val_r2"]))

    return pipeline, metrics


# ---------------------------------------------------------
# 6. TRAIN AND COMPARE OPTIMIZED MODELS
# ---------------------------------------------------------

results = {}

with mlflow.start_run(
    run_name="Optimized Model Comparison"
):

    mlflow.log_param(
        "training_records",
        len(X_train)
    )

    mlflow.log_param(
        "validation_records",
        len(X_val)
    )

    mlflow.log_param(
        "comparison_metric",
        "validation_rmse"
    )

    # -----------------------------------------------------
    # MODEL 1: OPTIMIZED RANDOM FOREST
    # -----------------------------------------------------

    print("\n\n========================================")
    print("OPTIMIZED RANDOM FOREST")
    print("========================================")

    rf_model = RandomForestRegressor(
        **rf_best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_pipeline = create_pipeline(
        model=rf_model,
        scale_numeric=False
    )

    with mlflow.start_run(
        run_name="Optimized Random Forest Final",
        nested=True
    ):

        mlflow.log_params(rf_best_params)

        rf_pipeline, rf_metrics = evaluate_model(
            "Optimized Random Forest",
            rf_pipeline,
            X_train,
            y_train,
            X_val,
            y_val
        )

        mlflow.log_metrics(rf_metrics)

        mlflow.set_tag(
            "model_type",
            "Random Forest"
        )

        mlflow.sklearn.log_model(
            sk_model=rf_pipeline,
            name="optimized_random_forest_pipeline",
            serialization_format="cloudpickle"
        )

        results["Optimized Random Forest"] = rf_metrics

    # -----------------------------------------------------
    # MODEL 2: OPTIMIZED XGBOOST
    # -----------------------------------------------------

    print("\n\n========================================")
    print("OPTIMIZED XGBOOST")
    print("========================================")

    xgb_model = XGBRegressor(
        **xgb_best_params,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_pipeline = create_pipeline(
        model=xgb_model,
        scale_numeric=False
    )

    with mlflow.start_run(
        run_name="Optimized XGBoost Final",
        nested=True
    ):

        mlflow.log_params(xgb_best_params)

        xgb_pipeline, xgb_metrics = evaluate_model(
            "Optimized XGBoost",
            xgb_pipeline,
            X_train,
            y_train,
            X_val,
            y_val
        )

        mlflow.log_metrics(xgb_metrics)

        mlflow.set_tag(
            "model_type",
            "XGBoost"
        )

        mlflow.sklearn.log_model(
            sk_model=xgb_pipeline,
            name="optimized_xgboost_pipeline",
            serialization_format="cloudpickle"
        )

        results["Optimized XGBoost"] = xgb_metrics

    # -----------------------------------------------------
    # 7. COMPARE BOTH MODELS
    # -----------------------------------------------------

    comparison_data = []

    for model_name, metrics in results.items():

        comparison_data.append({
            "Model": model_name,
            "Validation MAE": metrics["val_mae"],
            "Validation RMSE": metrics["val_rmse"],
            "Validation R2": metrics["val_r2"]
        })

    comparison_df = pd.DataFrame(comparison_data)

    comparison_df = comparison_df.sort_values(
        by="Validation RMSE",
        ascending=True
    ).reset_index(drop=True)

    print("\n\n========================================")
    print("OPTIMIZED MODEL COMPARISON")
    print("========================================")

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "Validation MAE": "{:.4f}".format,
                "Validation RMSE": "{:.4f}".format,
                "Validation R2": "{:.6f}".format
            }
        )
    )

    # -----------------------------------------------------
    # 8. IDENTIFY THE BEST MODEL
    # -----------------------------------------------------

    winner = comparison_df.iloc[0]

    winner_name = winner["Model"]

    winner_rmse = winner["Validation RMSE"]

    print("\n----------------------------------------")
    print("BEST MODEL")
    print("----------------------------------------")

    print("Model : {}".format(winner_name))

    print("Validation RMSE : {:.4f}".format(
        winner_rmse
    ))

    mlflow.set_tag(
        "selected_model",
        winner_name
    )

    mlflow.log_metric(
        "best_validation_rmse",
        winner_rmse
    )

    mlflow.log_metric(
        "best_validation_mae",
        winner["Validation MAE"]
    )

    mlflow.log_metric(
        "best_validation_r2",
        winner["Validation R2"]
    )

    print("\nBoth optimized models logged successfully to MLflow.")

    print("\nComparison completed successfully.")
