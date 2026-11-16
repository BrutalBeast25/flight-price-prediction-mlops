
import os
import pandas as pd


PROCESSED_DIR = os.path.join("data", "processed")

TRAIN_PATH = os.path.join(
    PROCESSED_DIR, "train.csv"
)

TEST_PATH = os.path.join(
    PROCESSED_DIR, "test.csv"
)


def engineer_features(df):

    df = df.copy()

    # 1. Directional route
    df["route"] = (
        df["source_city"] + "_to_" +
        df["destination_city"]
    )

    # 2. Airline and travel class interaction
    df["airline_class"] = (
        df["airline"] + "_" + df["class"]
    )

    # 3. Route and travel class interaction
    df["route_class"] = (
        df["route"] + "_" + df["class"]
    )

    # 4. Remove flight number initially
    if "flight" in df.columns:
        df = df.drop(columns=["flight"])

    return df


def main():

    print("Loading training and testing datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Training shape: {}".format(train_df.shape))
    print("Testing shape: {}".format(test_df.shape))

    # Apply identical feature engineering
    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)

    # Save engineered datasets
    train_features.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "train_features.csv"
        ),
        index=False
    )

    test_features.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "test_features.csv"
        ),
        index=False
    )

    print("\n========== FEATURE ENGINEERING SUMMARY ==========")

    print("Training shape: {}".format(train_features.shape))
    print("Testing shape: {}".format(test_features.shape))

    print("\nEngineered features:")
    print([
        "route",
        "airline_class",
        "route_class"
    ])

    print("\nFlight number removed: Yes")

    print("\nUnique routes: {}".format(
        train_features["route"].nunique()
    ))

    print("Unique airline-class combinations: {}".format(
        train_features["airline_class"].nunique()
    ))

    print("Unique route-class combinations: {}".format(
        train_features["route_class"].nunique()
    ))

    # Verify feature consistency
    if list(train_features.columns) == list(test_features.columns):
        print("\nFeature consistency: PASSED")
    else:
        print("\nFeature consistency: FAILED")

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()
