
import os
import json
import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_features.csv"
)

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "v1.0"
)

CATEGORICAL_PATH = os.path.join(
    ARTIFACT_DIR,
    "ui_options.json"
)

NUMERIC_PATH = os.path.join(
    ARTIFACT_DIR,
    "numeric_ranges.json"
)


# ---------------------------------------------------------
# 2. FEATURE DEFINITIONS
# ---------------------------------------------------------

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "class"
]

NUMERIC_FEATURES = [
    "duration",
    "days_left"
]


# ---------------------------------------------------------
# 3. LOAD TRAINING DATA
# ---------------------------------------------------------

print("\nLoading processed training dataset...")

data = pd.read_csv(TRAIN_DATA_PATH)

print("Training records: {}".format(len(data)))


# ---------------------------------------------------------
# 4. EXTRACT CATEGORICAL OPTIONS
# ---------------------------------------------------------

ui_options = {}

for column in CATEGORICAL_FEATURES:

    values = sorted(
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    ui_options[column] = values

    print(
        "{}: {} unique values".format(
            column,
            len(values)
        )
    )


# ---------------------------------------------------------
# 5. EXTRACT NUMERIC RANGES
# ---------------------------------------------------------

numeric_ranges = {}

for column in NUMERIC_FEATURES:

    numeric_ranges[column] = {
        "min": float(data[column].min()),
        "max": float(data[column].max()),
        "default": float(data[column].median())
    }

    print(
        "{} range: {} to {}".format(
            column,
            numeric_ranges[column]["min"],
            numeric_ranges[column]["max"]
        )
    )


# ---------------------------------------------------------
# 6. SAVE CONFIGURATION FILES
# ---------------------------------------------------------

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

with open(CATEGORICAL_PATH, "w") as file:

    json.dump(
        ui_options,
        file,
        indent=4
    )

with open(NUMERIC_PATH, "w") as file:

    json.dump(
        numeric_ranges,
        file,
        indent=4
    )


# ---------------------------------------------------------
# 7. COMPLETION
# ---------------------------------------------------------

print("\n========================================")
print("UI CONFIGURATION GENERATED")
print("========================================")

print("Categorical options: {}".format(CATEGORICAL_PATH))
print("Numeric ranges     : {}".format(NUMERIC_PATH))
