
import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from optuna.samplers import TPESampler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
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

OPTUNA_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "optuna_random_forest.db"
)

OPTUNA_STORAGE = "sqlite:///{}".format(
    OPTUNA_DB_PATH.replace("\\", "/")
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

print("\nTraining records: {}".format(len(X_train)))
print("Validation records: {}".format(len(X_val)))


# --------------------------------------------------
# 4. CROSS-VALIDATION CONFIGURATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Keep identical CV folds across every Optuna trial.
cv_splits = list(
    cv.split(X_train, X_train["class"])
)


# --------------------------------------------------
# 5. OPTUNA OBJECTIVE FUNCTION
# --------------------------------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 300, step=50
        ),

        "max_depth": trial.suggest_categorical(
            "max_depth", [10, 15, 20, 25, 30, None]
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 10
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 5
        ),

        "max_features": trial.suggest_float(
            "max_features", 0.5, 1.0
        )
    }

    model = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1
    )

    pipeline = create_pipeline(
        model=model,
        scale_numeric=False
    )

    # Track each Optuna trial separately in MLflow.
    with mlflow.start_run(
        run_name="RF Optuna Trial {}".format(trial.number),
        nested=True
    ):

        mlflow.log_params(params)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_splits,
            scoring="neg_root_mean_squared_error",
            n_jobs=1
        )

        cv_rmse = -scores.mean()

        mlflow.log_metric("cv_rmse", cv_rmse)

        mlflow.log_metric(
            "cv_rmse_std",
            scores.std()
        )

        print(
            "Trial {} | CV RMSE: {:.2f}".format(
                trial.number,
                cv_rmse
            )
        )

    return cv_rmse


# --------------------------------------------------
# 6. RUN OPTUNA OPTIMIZATION
# --------------------------------------------------

study = optuna.create_study(
    study_name="random_forest_optimization",
    storage=OPTUNA_STORAGE,
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=42)
)

print("\nStarting Random Forest optimization...")

with mlflow.start_run(
    run_name="Random Forest Optuna Optimization"
):

    mlflow.log_param("optimization", "Optuna TPE")
    mlflow.log_param("optimization_trials", 15)
    mlflow.log_param("cross_validation_folds", 3)
    mlflow.log_param("optimization_metric", "RMSE")

    study.optimize(
        objective,
        n_trials=15
    )

    print("\nOptimization completed.")

    print("\nBest Trial:")
    print("Trial Number: {}".format(study.best_trial.number))
    print("Best CV RMSE: {:.2f}".format(study.best_value))

    print("\nBest Hyperparameters:")

    for key, value in study.best_params.items():
        print("{}: {}".format(key, value))

    # --------------------------------------------------
    # 7. TRAIN BEST MODEL
    # --------------------------------------------------

    best_model = RandomForestRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )

    best_pipeline = create_pipeline(
        model=best_model,
        scale_numeric=False
    )

    print("\nTraining optimized Random Forest...")

    best_pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # 8. VALIDATION EVALUATION
    # --------------------------------------------------

    train_predictions = best_pipeline.predict(X_train)
    val_predictions = best_pipeline.predict(X_val)

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

    print("\nOptimized Random Forest Results")
    print("--------------------------------")

    print("Training MAE: {:.2f}".format(train_mae))
    print("Training RMSE: {:.2f}".format(train_rmse))
    print("Training R2: {:.4f}".format(train_r2))

    print("\nValidation Results")

    print("Validation MAE: {:.2f}".format(val_mae))
    print("Validation RMSE: {:.2f}".format(val_rmse))
    print("Validation R2: {:.4f}".format(val_r2))

    # --------------------------------------------------
    # 9. MLFLOW LOGGING
    # --------------------------------------------------

    mlflow.log_params({
        "best_trial_number": study.best_trial.number,
        "best_cv_rmse": study.best_value,
        "training_records": len(X_train),
        "validation_records": len(X_val),
        "total_records": len(df)
    })

    mlflow.log_params({
        "best_{}".format(key): value
        for key, value in study.best_params.items()
    })

    mlflow.log_metrics({
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2
    })

    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="optimized_random_forest",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("\nOptimized model successfully logged to MLflow.")
