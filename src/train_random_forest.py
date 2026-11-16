
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

    print("\n========== RANDOM FOREST MODEL TRAINING ==========")

    # --------------------------------------------------------
    # Configure MLflow
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
    # Create Random Forest model
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Create preprocessing + model pipeline
    # --------------------------------------------------------

    pipeline = create_pipeline(
        model=model,
        scale_numeric=False
    )

    # --------------------------------------------------------
    # Start MLflow experiment run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="Random_Forest_Regressor"
    ) as run:

        print("\nTraining Random Forest Regressor...")

        # Train using training dataset only
        pipeline.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        print("\nGenerating predictions...")

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
            "model_type": "RandomForestRegressor",
            "preprocessing": "OneHotEncoder",
            "numerical_scaling": False,
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "training_records": len(X_train),
            "validation_records": len(X_val),
            "input_features": X_train.shape[1],
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

        # mlflow.sklearn.log_model(
        #     sk_model=pipeline,
        #     artifact_path="flight_price_model"
        # )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="flight_price_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
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

        print("\nRandom Forest model logged successfully.")

    print("\nTraining completed successfully!")


# ============================================================
# 4. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
