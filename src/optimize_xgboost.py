
import os
import warnings

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from optuna.samplers import TPESampler
from optuna.trial import TrialState

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

OPTUNA_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "optuna_xgboost.db"
)

MLFLOW_TRACKING_URI = "sqlite:///{}".format(
    MLFLOW_DB_PATH.replace("\\", "/")
)

OPTUNA_STORAGE = "sqlite:///{}".format(
    OPTUNA_DB_PATH.replace("\\", "/")
)

EXPERIMENT_NAME = "Flight Price Prediction"
STUDY_NAME = "xgboost_optimization"

TARGET_COLUMN = "price"

N_TRIALS = 10
N_SPLITS = 3
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. LOAD DATA AND CREATE TRAIN/VALIDATION SPLIT
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
# 3. CREATE FIXED CROSS-VALIDATION FOLDS
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Reuse exactly the same folds for every Optuna trial.
cv_splits = list(
    cv.split(X_train, X_train["class"])
)


# ---------------------------------------------------------
# 4. CONFIGURE MLFLOW
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# 5. DEFINE OPTUNA OBJECTIVE
# ---------------------------------------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 700, step=100
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.20, log=True
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.65, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.60, 1.0
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),

        "gamma": trial.suggest_float(
            "gamma", 0.0, 5.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 20.0, log=True
        )
    }

    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = create_pipeline(
        model=model,
        scale_numeric=False
    )

    # Each trial gets its own nested MLflow run.
    with mlflow.start_run(
        run_name="XGBoost Trial {}".format(trial.number),
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
        cv_rmse_std = scores.std()

        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_rmse_std", cv_rmse_std)

        trial.set_user_attr(
            "cv_rmse_std",
            float(cv_rmse_std)
        )

        print(
            "\nTrial {} | CV RMSE: {:.4f} | Std: {:.4f}".format(
                trial.number,
                cv_rmse,
                cv_rmse_std
            )
        )

        return cv_rmse


# ---------------------------------------------------------
# 6. CREATE OR RESUME OPTUNA STUDY
# ---------------------------------------------------------

sampler = TPESampler(
    seed=RANDOM_STATE
)

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=OPTUNA_STORAGE,
    sampler=sampler,
    direction="minimize",
    load_if_exists=True
)


# ---------------------------------------------------------
# 7. RUN OPTIMIZATION WITH INTERRUPTION RECOVERY
# ---------------------------------------------------------

completed_trials = study.get_trials(
    states=(TrialState.COMPLETE,)
)

remaining_trials = max(
    0,
    N_TRIALS - len(completed_trials)
)

print("\nExisting completed trials : {}".format(
    len(completed_trials)
))

print("Target completed trials   : {}".format(
    N_TRIALS
))

print("Remaining trials           : {}".format(
    remaining_trials
))


with mlflow.start_run(
    run_name="XGBoost Optuna Optimization"
):

    mlflow.log_param("optimization_algorithm", "Optuna TPE")
    mlflow.log_param("target_completed_trials", N_TRIALS)
    mlflow.log_param("cv_folds", N_SPLITS)
    mlflow.log_param("random_state", RANDOM_STATE)

    try:

        if remaining_trials > 0:

            print("\nStarting XGBoost optimization...")

            study.optimize(
                objective,
                n_trials=remaining_trials
            )

        else:

            print("\nTarget number of completed trials already reached.")

    except KeyboardInterrupt:

        print("\n\nOptimization interrupted by user.")
        print("Recovering completed Optuna trials...")

        mlflow.set_tag(
            "optimization_status",
            "interrupted"
        )

    except Exception as error:

        print("\nOptimization stopped due to an error.")
        print("Error: {}".format(str(error)))

        mlflow.set_tag(
            "optimization_status",
            "error"
        )

        mlflow.set_tag(
            "optimization_error",
            str(error)[:450]
        )

    # -----------------------------------------------------
    # 8. RECOVER THE BEST COMPLETED TRIAL
    # -----------------------------------------------------

    completed_trials = study.get_trials(
        states=(TrialState.COMPLETE,)
    )

    print("\nCompleted trials available: {}".format(
        len(completed_trials)
    ))

    if len(completed_trials) == 0:

        print(
            "\nNo completed trials are available yet. "
            "Run this script again to resume optimization."
        )

    else:

        best_trial = study.best_trial
        best_params = study.best_params
        best_cv_rmse = study.best_value

        print("\n----------------------------------------")
        print("BEST XGBOOST OPTUNA TRIAL")
        print("----------------------------------------")

        print("Best Trial Number : {}".format(
            best_trial.number
        ))

        print("Best CV RMSE      : {:.4f}".format(
            best_cv_rmse
        ))

        print("\nBest Parameters:")

        for key, value in best_params.items():

            print("{} : {}".format(key, value))

            mlflow.log_param(
                "best_{}".format(key),
                value
            )

        mlflow.log_metric(
            "best_cv_rmse",
            best_cv_rmse
        )

        # -------------------------------------------------
        # 9. TRAIN BEST MODEL ON COMPLETE TRAINING PARTITION
        # -------------------------------------------------

        print("\nTraining optimized XGBoost model...")

        best_model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        best_pipeline = create_pipeline(
            model=best_model,
            scale_numeric=False
        )

        best_pipeline.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # 10. EVALUATE TRAINING AND VALIDATION PERFORMANCE
        # -------------------------------------------------

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

        print("\n----------------------------------------")
        print("OPTIMIZED XGBOOST RESULTS")
        print("----------------------------------------")

        print("\nTraining Metrics")

        print("MAE  : {:.4f}".format(train_mae))
        print("RMSE : {:.4f}".format(train_rmse))
        print("R2   : {:.6f}".format(train_r2))

        print("\nValidation Metrics")

        print("MAE  : {:.4f}".format(val_mae))
        print("RMSE : {:.4f}".format(val_rmse))
        print("R2   : {:.6f}".format(val_r2))

        # -------------------------------------------------
        # 11. LOG FINAL MODEL AND METRICS TO MLFLOW
        # -------------------------------------------------

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        mlflow.log_param(
            "completed_optuna_trials",
            len(completed_trials)
        )

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="optimized_xgboost_pipeline",
            serialization_format="cloudpickle"
        )

        mlflow.set_tag(
            "optimization_status",
            "completed"
            if len(completed_trials) >= N_TRIALS
            else "partial"
        )

        print("\nOptimized model logged successfully to MLflow.")

        print("\nOptuna database:")
        print(OPTUNA_DB_PATH)

        print("\nOptimization and model evaluation completed.")
