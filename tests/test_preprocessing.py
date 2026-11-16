
import os
import sys
import pandas as pd

from sklearn.base import clone

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    create_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# Dataset paths
TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test_features.csv"
)


def main():

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Training shape: {}".format(train_df.shape))
    print("Testing shape: {}".format(test_df.shape))

    # Separate features and target
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]

    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    print("\n========== FEATURE VALIDATION ==========")

    expected_features = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    missing_train = [
        col for col in expected_features
        if col not in X_train.columns
    ]

    missing_test = [
        col for col in expected_features
        if col not in X_test.columns
    ]

    if missing_train or missing_test:
        raise ValueError(
            "Missing features. Train: {} | Test: {}".format(
                missing_train,
                missing_test
            )
        )

    print("All expected features are available.")

    # Create preprocessing object
    print("\n========== PREPROCESSING ==========")

    preprocessor = create_preprocessor(
        scale_numeric=True
    )

    # Fit only on training data
    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data using the fitted preprocessor
    X_test_transformed = preprocessor.transform(
        X_test
    )

    print("Training transformed shape: {}".format(
        X_train_transformed.shape
    ))

    print("Testing transformed shape: {}".format(
        X_test_transformed.shape
    ))

    # Verify dimensions
    if X_train_transformed.shape[1] != X_test_transformed.shape[1]:
        raise ValueError(
            "Training and testing feature dimensions differ."
        )

    print("\nFeature dimensions: PASSED")

    # Check numerical scaling
    numerical_scaler = preprocessor.named_transformers_[
        "numerical"
    ]

    print("\n========== NUMERICAL SCALING ==========")

    print("Scaler means: {}".format(
        numerical_scaler.mean_
    ))

    print("Scaler standard deviations: {}".format(
        numerical_scaler.scale_
    ))

    # Check categorical encoding
    categorical_encoder = preprocessor.named_transformers_[
        "categorical"
    ]

    encoded_features = (
        categorical_encoder.get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )

    print("\n========== CATEGORICAL ENCODING ==========")

    print("Encoded categorical features: {}".format(
        len(encoded_features)
    ))

    print("Total transformed features: {}".format(
        X_train_transformed.shape[1]
    ))

    # Check transformed values
    print("\n========== DATA VALIDATION ==========")

    if hasattr(X_train_transformed, "toarray"):
        train_array = X_train_transformed.toarray()
    else:
        train_array = X_train_transformed

    if hasattr(X_test_transformed, "toarray"):
        test_array = X_test_transformed.toarray()
    else:
        test_array = X_test_transformed

    import numpy as np

    if not np.isfinite(train_array).all():
        raise ValueError("Invalid values found in training output.")

    if not np.isfinite(test_array).all():
        raise ValueError("Invalid values found in testing output.")

    print("No NaN or infinite values found.")

    print("\nTarget shapes:")
    print("y_train: {}".format(y_train.shape))
    print("y_test: {}".format(y_test.shape))

    print("\nPreprocessing pipeline test completed successfully!")


if __name__ == "__main__":
    main()
