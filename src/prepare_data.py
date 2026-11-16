
import os
import pandas as pd

RAW_PATH = os.path.join(
    "data", "raw", "Flight_Dataset.xlsx"
)

PROCESSED_DIR = os.path.join(
    "data", "processed"
)

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR, "flight_features.csv"
)


def main():

    print("Loading raw dataset...")

    df = pd.read_excel(RAW_PATH)

    print("Original shape: {}".format(df.shape))

    # Create directional route feature
    df["route"] = (
        df["source_city"] + "_to_" +
        df["destination_city"]
    )

    # Initial feature selection
    selected_columns = [
        "airline",
        "source_city",
        "destination_city",
        "route",
        "departure_time",
        "arrival_time",
        "stops",
        "class",
        "duration",
        "days_left",
        "price"
    ]

    model_df = df[selected_columns].copy()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    model_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== PREPARED DATASET ==========")

    print("Shape: {}".format(model_df.shape))

    print("\nColumns:")
    print(model_df.columns.tolist())

    print("\nMissing values:")
    print(model_df.isnull().sum())

    print("\nRoute count: {}".format(
        model_df["route"].nunique()
    ))

    print("\nSaved to: {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
