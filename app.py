from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    FEATURE_COLUMNS,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_DISPLAY_NAMES,
    MODEL_FILENAMES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    ROOT_TEST_DATA_FILE,
    TARGET_COLUMN,
)
from src.data_validation import standardize_dataset
from src.evaluate_models import evaluate_model, evaluate_models


EXPECTED_UPLOAD_FILENAME = "test_data.csv"
INVALID_FILENAME_ERROR = (
    "Invalid file uploaded! Please upload the original test_data.csv file provided with this project. "
    "Other files are not accepted."
)


st.set_page_config(
    page_title="Online Shopper Purchase Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        opacity: 1 !important;
        visibility: visible !important;
        transform-origin: left center;
    }

    [data-testid="collapsedControl"] {
        margin: 0.75rem !important;
        min-width: 58px !important;
        min-height: 58px !important;
        width: 58px !important;
        height: 58px !important;
        background: #f2f5fa !important;
        border: 1.5px solid #a9b4c2 !important;
        border-radius: 14px !important;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.28) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }

    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button,
    button[title="Open sidebar"],
    button[aria-label="Open sidebar"],
    button[title="Close sidebar"],
    button[aria-label="Close sidebar"] {
        min-width: 56px !important;
        min-height: 56px !important;
        background: #f2f5fa !important;
        border: 1.5px solid #a9b4c2 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="collapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    button[title="Open sidebar"]:hover,
    button[aria-label="Open sidebar"]:hover,
    button[title="Close sidebar"]:hover,
    button[aria-label="Close sidebar"]:hover {
        background: #e7edf6 !important;
        border-color: #7f8da1 !important;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.30) !important;
    }

    [data-testid="collapsedControl"] > * {
        transform: scale(1.45) !important;
    }

    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    button[title="Open sidebar"] svg,
    button[aria-label="Open sidebar"] svg,
    button[title="Close sidebar"] svg,
    button[aria-label="Close sidebar"] svg {
        width: 1.9rem !important;
        height: 1.9rem !important;
        color: #5f6f83 !important;
        fill: #5f6f83 !important;
        stroke: #5f6f83 !important;
    }

    .app-subtitle {
        font-size: 1.25rem;
        font-weight: 600;
        color: #44546a;
        margin-top: -0.35rem;
        margin-bottom: 0.4rem;
    }

    .landing-callout {
        background: #eef6ff;
        border-left: 5px solid #2c7be5;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        font-weight: 600;
        color: #17324d;
        margin-top: 1.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metadata() -> dict[str, Any]:
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


@st.cache_data
def load_reference_metrics() -> pd.DataFrame:
    if METRICS_FILE.exists():
        return pd.read_csv(METRICS_FILE)
    return pd.DataFrame()


@st.cache_data
def load_expected_test_data() -> pd.DataFrame:
    return standardize_dataset(pd.read_csv(ROOT_TEST_DATA_FILE))


@st.cache_resource
def load_models() -> dict[str, Any]:
    models = {}
    for model_key, filename in MODEL_FILENAMES.items():
        display_name = MODEL_DISPLAY_NAMES[model_key]
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing saved model artifact: {model_path}")
        models[display_name] = joblib.load(model_path)
    return models


def validate_uploaded_filename(filename: str) -> str | None:
    if filename != EXPECTED_UPLOAD_FILENAME:
        return INVALID_FILENAME_ERROR
    return None


def normalize_for_test_data_comparison(data: pd.DataFrame) -> pd.DataFrame:
    data = standardize_dataset(data.copy())
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    data = data[required_columns].copy()

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    categorical_and_target_columns = [
        column for column in required_columns if column not in NUMERIC_FEATURES
    ]
    for column in categorical_and_target_columns:
        data[column] = data[column].astype(str)

    return data.reset_index(drop=True)


def uploaded_matches_expected_test_data(uploaded_data: pd.DataFrame, expected_data: pd.DataFrame) -> bool:
    try:
        normalized_uploaded = normalize_for_test_data_comparison(uploaded_data)
        normalized_expected = normalize_for_test_data_comparison(expected_data)
    except (KeyError, ValueError):
        return False

    if normalized_uploaded.shape != normalized_expected.shape:
        return False

    return normalized_uploaded.equals(normalized_expected)


def validate_uploaded_data(
    data: pd.DataFrame,
    metadata: dict[str, Any],
    expected_test_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if data.empty:
        errors.append("The uploaded CSV is empty.")
        return None, errors, warnings

    target_before_standardization = data[TARGET_COLUMN].copy() if TARGET_COLUMN in data.columns else None
    data = standardize_dataset(data)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    unexpected_columns = [column for column in data.columns if column not in required_columns]

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if unexpected_columns:
        warnings.append(f"Unexpected columns will be ignored: {', '.join(unexpected_columns)}")

    if TARGET_COLUMN in data.columns:
        invalid_target_values = []
        if target_before_standardization is not None:
            invalid_mask = data[TARGET_COLUMN].isna() & target_before_standardization.notna()
            invalid_target_values = sorted(target_before_standardization[invalid_mask].astype(str).unique().tolist())
        if invalid_target_values:
            errors.append(f"Unexpected target label(s): {', '.join(invalid_target_values)}")

        missing_target_mask = data[TARGET_COLUMN].isna()
        if target_before_standardization is not None:
            missing_target_mask = missing_target_mask & target_before_standardization.isna()
        if missing_target_mask.any():
            errors.append("The target column contains missing values.")

    for column in NUMERIC_FEATURES:
        if column in data.columns:
            converted = pd.to_numeric(data[column], errors="coerce")
            invalid_count = int(converted.isna().sum() - data[column].isna().sum())
            if invalid_count > 0:
                errors.append(f"Column `{column}` has {invalid_count} non-numeric value(s).")
            data[column] = converted

    observed_categories = metadata.get("categorical_values_observed_in_training_dataset", {})
    for column, allowed_values in observed_categories.items():
        if column in data.columns:
            uploaded_values = set(data[column].dropna().astype(str).unique())
            unseen_values = sorted(uploaded_values - set(allowed_values))
            if unseen_values:
                warnings.append(
                    f"Column `{column}` has unseen value(s): {', '.join(unseen_values)}. "
                    "The saved encoder will handle them as unknown categories."
                )

    allowed_targets = set(metadata.get("target_classes", []))
    if TARGET_COLUMN in data.columns:
        uploaded_targets = set(data[TARGET_COLUMN].dropna().astype(str).unique())
        invalid_targets = sorted(uploaded_targets - allowed_targets)
        if invalid_targets:
            errors.append(f"Unexpected target label(s): {', '.join(invalid_targets)}")

    if errors:
        return None, errors, warnings

    if expected_test_data is not None and not uploaded_matches_expected_test_data(data, expected_test_data):
        errors.append(
            "The uploaded file is named test_data.csv, but its contents do not match the original "
            "test_data.csv file provided with this project."
        )
        return None, errors, warnings

    return data[required_columns], errors, warnings


def plot_confusion_matrix(y_true: pd.Series, y_pred: list[str], labels: list[str]):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    return fig


def render_metrics(row: pd.Series) -> None:
    columns = st.columns(6)
    for column, metric_name in zip(columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        column.metric(metric_name, f"{row[metric_name]:.4f}")


def render_landing_screen() -> None:
    st.markdown("### How to Use This Application")
    st.markdown(
        """
        1. Upload the original `test_data.csv` file using the sidebar and select an individual classifier.
        2. Navigate to **Evaluate Upload** and examine its performance.
        3. Navigate to **Compare All Models** to compare all five classifiers side by side.
        """
    )

    st.markdown(
        """
        <div class="landing-callout">
            Get started by uploading the original test_data.csv file from the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Online Shopper Purchase Intention Classifier")
    st.markdown(
        '<div class="app-subtitle">Machine Learning Model Evaluation & Comparison Dashboard</div>',
        unsafe_allow_html=True,
    )

    if not METADATA_FILE.exists():
        st.error("Project metadata is missing. Run `py -m src.train_models` from the project root first.")
        st.stop()

    metadata = load_metadata()
    models = load_models()

    with st.sidebar:
        st.caption("Close sidebar")
        st.header("Evaluation Controls")
        selected_model = st.selectbox(
            "Choose a classifier",
            list(MODEL_DISPLAY_NAMES.values()),
        )
        st.write("Upload the original test_data.csv file to evaluate the saved classifiers.")
        uploaded_file = st.file_uploader("Upload labeled test CSV", type=["csv"], key="sidebar_upload")
        st.divider()
        st.markdown(
            """
            - Only `test_data.csv` is accepted.
            """
        )
        st.divider()
        st.write("Expected target column:", f"`{TARGET_COLUMN}`")
        st.write("Expected feature count:", len(FEATURE_COLUMNS))

    if uploaded_file is None:
        render_landing_screen()
        reference_metrics = load_reference_metrics()
        if not reference_metrics.empty:
            st.subheader("Reference Metrics from Held-Out Test Data")
            st.dataframe(reference_metrics, use_container_width=True, hide_index=True)
        return

    filename_error = validate_uploaded_filename(uploaded_file.name)
    if filename_error:
        st.error(filename_error)
        st.stop()

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

    expected_test_data = load_expected_test_data()
    validated_data, errors, warnings = validate_uploaded_data(uploaded_data, metadata, expected_test_data)
    for warning in warnings:
        st.warning(warning)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    assert validated_data is not None
    x_uploaded = validated_data[FEATURE_COLUMNS]
    y_uploaded = validated_data[TARGET_COLUMN]

    tab_evaluate, tab_compare = st.tabs(["Evaluate Upload", "Compare All Models"])

    with tab_evaluate:
        st.subheader("Uploaded Dataset")
        dim_col, class_col = st.columns([0.7, 2.3])
        dim_col.metric("Rows", validated_data.shape[0])
        dim_col.metric("Columns", validated_data.shape[1])
        class_col.bar_chart(y_uploaded.value_counts(), height=220)

        st.subheader(f"Selected Model: {selected_model}")
        model = models[selected_model]
        metric_row, details = evaluate_model(model, x_uploaded, y_uploaded, model_name=selected_model)
        render_metrics(pd.Series(metric_row))

        labels = details["class_labels"]
        predictions = details["predictions"]
        cm_col, report_col = st.columns([1, 1])
        with cm_col:
            st.pyplot(plot_confusion_matrix(y_uploaded, predictions, labels))
            st.caption("Rows are actual labels and columns are predicted labels.")
        with report_col:
            report = classification_report(
                y_uploaded,
                predictions,
                labels=labels,
                zero_division=0,
                output_dict=True,
            )
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    with tab_compare:
        st.subheader("All Five Models on Uploaded Test Data")
        comparison_metrics, _ = evaluate_models(models, x_uploaded, y_uploaded)
        st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)

        chart_data = comparison_metrics.set_index("ML Model Name")[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
