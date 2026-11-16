
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Feature definitions
NUMERICAL_FEATURES = [
    "duration",
    "days_left"
]

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "route",
    "departure_time",
    "arrival_time",
    "stops",
    "class",
    "airline_class",
    "route_class"
]


def create_preprocessor(scale_numeric=True):

    # Numerical preprocessing
    if scale_numeric:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"

    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


def create_pipeline(model, scale_numeric=True):

    preprocessor = create_preprocessor(
        scale_numeric=scale_numeric
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline
