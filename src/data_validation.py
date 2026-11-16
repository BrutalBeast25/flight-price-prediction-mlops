
import os
import re
import pandas as pd

RAW_DIR = os.path.join("data", "raw")

EXPECTED_COLUMNS = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "price"
]


def normalize_column(column):
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


def main():

    files = [
        file for file in os.listdir(RAW_DIR)
        if file.lower().endswith((".xlsx", ".xls", ".csv"))
    ]

    if len(files) != 1:
        raise ValueError(
            "Expected exactly one dataset file in {}. Found: {}".format(
                RAW_DIR, files
            )
        )

    file_path = os.path.join(RAW_DIR, files[0])

    print("\nLoading dataset: {}".format(file_path))

    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        excel = pd.ExcelFile(file_path)

        print("\nAvailable Excel sheets:")
        print(excel.sheet_names)

        df = pd.read_excel(file_path, sheet_name=0)

    df.columns = [
        normalize_column(column) for column in df.columns
    ]

    print("\n========== DATASET OVERVIEW ==========")

    print("Rows: {}".format(df.shape[0]))
    print("Columns: {}".format(df.shape[1]))

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\nTotal Missing Values: {}".format(
        df.isnull().sum().sum()
    ))

    print("\n========== DUPLICATE RECORDS ==========")

    print("Duplicate Rows: {}".format(
        df.duplicated().sum()
    ))

    print("\n========== NUMERIC SUMMARY ==========")
    print(df.describe())

    print("\n========== CATEGORICAL SUMMARY ==========")

    print(df.describe(include="object"))

    print("\n========== EXPECTED COLUMN CHECK ==========")

    actual_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    print("Missing Expected Columns:")
    print(sorted(expected_columns - actual_columns))

    print("Additional Columns:")
    print(sorted(actual_columns - expected_columns))

    if "price" in df.columns:

        print("\n========== TARGET VARIABLE: PRICE ==========")

        print(df["price"].describe())

        print("\nNegative Prices: {}".format(
            (df["price"] < 0).sum()
        ))

    print("\n========== SAMPLE RECORDS ==========")
    print(df.head())

    print("\nDataset validation inspection completed.")


if __name__ == "__main__":
    main()
