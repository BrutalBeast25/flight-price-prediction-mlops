
import os
import mlflow
import mlflow.sklearn
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from src.preprocessing import create_pipeline


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

MLFLOW_TRACKING_URI = "sqlite:///{}".format(
    MLFLOW_DB_PATH.replace("\\", "/")
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Flight Price Prediction")


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["price"])
y = df["price"]

print("Dataset loaded successfully")
print("Total records: {}".format(len(df)))
print("Total features: {}".format(X.shape[1]))


# --------------------------------------------------
# 3. TRAIN-VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=X["class"]
)

print("\nDataset split completed")
print("Training records: {}".format(len(X_train)))
print("Validation records: {}".format(len(X_val)))
print("Total split records: {}".format(
    len(X_train) + len(X_val)
))


# --------------------------------------------------
# 4. XGBOOST MODEL
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. PREPROCESSING PIPELINE
# --------------------------------------------------

pipeline = create_pipeline(
    model=model,
    scale_numeric=False
)


# --------------------------------------------------
# 6. MODEL TRAINING AND EVALUATION
# --------------------------------------------------

with mlflow.start_run(run_name="XGBoost Regressor"):

    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    val_predictions = pipeline.predict(X_val)

    # Training metrics
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

    # Validation metrics
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

    # Display results
    print("\nXGBoost Training Results")
    print("------------------------")

    print("Training MAE: {}".format(train_mae))
    print("Training RMSE: {}".format(train_rmse))
    print("Training R2: {}".format(train_r2))

    print("\nValidation Results")
    print("------------------")

    print("Validation MAE: {}".format(val_mae))
    print("Validation RMSE: {}".format(val_rmse))
    print("Validation R2: {}".format(val_r2))

    # --------------------------------------------------
    # 7. MLFLOW LOGGING
    # --------------------------------------------------

    mlflow.log_params({
        "model": "XGBoost Regressor",
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "test_size": 0.20,
        "split_random_state": 42,
        "scaling": False,
        "training_records": len(X_train),
        "validation_records": len(X_val),
        "total_records": len(df),
        "n_features": X.shape[1]
    })

    mlflow.log_metrics({
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2
    })

    # Log complete preprocessing + model pipeline
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="flight_price_xgboost",
        serialization_format="cloudpickle"
    )

    print("\nXGBoost model logged successfully to MLflow.")
