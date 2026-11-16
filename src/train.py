
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.preprocessing import create_pipeline


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

# SQLite MLflow tracking database
MLFLOW_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

MLFLOW_TRACKING_URI = "sqlite:///{}".format(
    MLFLOW_DB_PATH.replace("\\", "/")
)


# ============================================================
# 2. METRIC CALCULATION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# 3. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n========== FLIGHT PRICE MODEL TRAINING ==========")

    # --------------------------------------------------------
    # Configure MLflow with SQLite
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        "Flight Price Prediction"
    )

    print("\nMLflow tracking URI: {}".format(
        mlflow.get_tracking_uri()
    ))

    print("MLflow database: {}".format(
        MLFLOW_DB_PATH
    ))

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    X = train_df.drop(
        columns=["price"]
    )

    y = train_df["price"]

    print("Total training dataset records: {}".format(
        len(X)
    ))

    print("Input features: {}".format(
        X.shape[1]
    ))

    # --------------------------------------------------------
    # Split into training and validation datasets
    # --------------------------------------------------------

    print("\nSplitting training and validation datasets...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=X["class"]
    )

    print("Model training records: {}".format(
        len(X_train)
    ))

    print("Validation records: {}".format(
        len(X_val)
    ))

    # --------------------------------------------------------
    # Create preprocessing + model pipeline
    # --------------------------------------------------------

    model = LinearRegression()

    pipeline = create_pipeline(
        model=model,
        scale_numeric=True
    )

    # --------------------------------------------------------
    # Start MLflow experiment
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="Linear_Regression_Baseline"
    ) as run:

        print("\nTraining Linear Regression...")

        # Train using training dataset only
        pipeline.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        train_predictions = pipeline.predict(
            X_train
        )

        val_predictions = pipeline.predict(
            X_val
        )

        # ----------------------------------------------------
        # Calculate evaluation metrics
        # ----------------------------------------------------

        train_metrics = calculate_metrics(
            y_train,
            train_predictions
        )

        val_metrics = calculate_metrics(
            y_val,
            val_predictions
        )

        # ----------------------------------------------------
        # Log experiment parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "model_type": "LinearRegression",
            "preprocessing": "OneHotEncoder + StandardScaler",
            "numerical_scaling": True,
            "training_records": len(X_train),
            "validation_records": len(X_val),
            "input_features": X_train.shape[1],
            "random_state": 42,
            "validation_size": 0.20
        })

        # ----------------------------------------------------
        # Log evaluation metrics
        # ----------------------------------------------------

        mlflow.log_metrics({
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "validation_mae": val_metrics["mae"],
            "validation_rmse": val_metrics["rmse"],
            "validation_r2": val_metrics["r2"]
        })

        # ----------------------------------------------------
        # Log trained model pipeline
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="flight_price_model"
        )

        # ----------------------------------------------------
        # Display training results
        # ----------------------------------------------------

        print("\n========== TRAINING RESULTS ==========")

        print("\nTraining Performance")

        print("MAE: Rs. {:.2f}".format(
            train_metrics["mae"]
        ))

        print("RMSE: Rs. {:.2f}".format(
            train_metrics["rmse"]
        ))

        print("R2 Score: {:.4f}".format(
            train_metrics["r2"]
        ))

        print("\nValidation Performance")

        print("MAE: Rs. {:.2f}".format(
            val_metrics["mae"]
        ))

        print("RMSE: Rs. {:.2f}".format(
            val_metrics["rmse"]
        ))

        print("R2 Score: {:.4f}".format(
            val_metrics["r2"]
        ))

        print("\nMLflow Run ID: {}".format(
            run.info.run_id
        ))

        print("\nModel pipeline logged successfully.")

    print("\nTraining completed successfully!")


# ============================================================
# 4. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
