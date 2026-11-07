from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_DOWNLOAD_URL,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)


def download_raw_dataset() -> pd.DataFrame:
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILE.exists():
        return load_raw_dataset()

    with urllib.request.urlopen(DATASET_DOWNLOAD_URL, timeout=60) as response:
        zipped_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as archive:
        with archive.open("online_shoppers_intention.csv") as csv_file:
            data = pd.read_csv(csv_file)

    data.to_csv(RAW_DATA_FILE, index=False)
    return standardize_dataset(data)


def standardize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if TARGET_COLUMN in data.columns:
        data[TARGET_COLUMN] = data[TARGET_COLUMN].map(
            {
                True: "TRUE",
                False: "FALSE",
                "TRUE": "TRUE",
                "FALSE": "FALSE",
                "True": "TRUE",
                "False": "FALSE",
            }
        )
    return data


def load_raw_dataset() -> pd.DataFrame:
    return standardize_dataset(pd.read_csv(RAW_DATA_FILE))


def validate_assignment_requirements(data: pd.DataFrame) -> dict:
    missing_required = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    duplicate_count = int(data.duplicated().sum())
    constant_features = [
        column for column in FEATURE_COLUMNS if column in data.columns and data[column].nunique(dropna=False) <= 1
    ]

    summary = {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "input_feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "missing_required_columns": missing_required,
        "missing_values_by_column": data.isna().sum().astype(int).to_dict(),
        "duplicate_records": duplicate_count,
        "target_class_distribution": data[TARGET_COLUMN].value_counts().astype(int).to_dict(),
        "constant_or_unusable_features": constant_features,
        "identifier_columns_removed": [],
        "leakage_review": {
            "identifier_columns": "No unique session/user identifier is present in the released CSV.",
            "page_values_note": (
                "PageValues is retained because it is a documented Google Analytics session feature, "
                "but it is expected to be strongly predictive and is discussed in the observations."
            ),
        },
        "meets_minimum_instances_500": data.shape[0] >= 500,
        "meets_minimum_features_12": len(FEATURE_COLUMNS) >= 12,
    }
    return summary


def save_json(data: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
