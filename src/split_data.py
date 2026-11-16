
import os
import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_PATH = os.path.join(
    "data", "processed", "flight_features.csv"
)

OUTPUT_DIR = os.path.join(
    "data", "processed"
)


def main():

    print("Loading prepared dataset...")

    df = pd.read_csv(INPUT_PATH)

    print("Dataset shape: {}".format(df.shape))

    # Separate features and target
    X = df.drop(columns=["price"])
    y = df["price"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=df["class"]
    )

    # Reconstruct train and test datasets
    train_df = X_train.copy()
    train_df["price"] = y_train

    test_df = X_test.copy()
    test_df["price"] = y_test

    # Save datasets
    train_df.to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(OUTPUT_DIR, "test.csv"),
        index=False
    )

    print("\n========== DATA SPLIT SUMMARY ==========")

    print("Original records: {}".format(len(df)))
    print("Training records: {}".format(len(train_df)))
    print("Testing records: {}".format(len(test_df)))

    print("\nTraining class distribution:")
    print(train_df["class"].value_counts(normalize=True))

    print("\nTesting class distribution:")
    print(test_df["class"].value_counts(normalize=True))

    print("\nTraining dataset saved: data/processed/train.csv")
    print("Testing dataset saved: data/processed/test.csv")


if __name__ == "__main__":
    main()
