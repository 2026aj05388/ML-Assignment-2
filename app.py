from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    EVALUATION_RESULTS_FILE,
    FEATURE_COLUMNS,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_DISPLAY_NAMES,
    MODEL_FILENAMES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.data_validation import standardize_dataset
from src.evaluate_models import evaluate_model, evaluate_models


st.set_page_config(
    page_title="Online Shopper Purchase Classifier",
    layout="wide",
)


@st.cache_data
def load_metadata() -> dict[str, Any]:
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


@st.cache_data
def load_reference_metrics() -> pd.DataFrame:
    if METRICS_FILE.exists():
        return pd.read_csv(METRICS_FILE)
    return pd.DataFrame()


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


def validate_uploaded_data(data: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if data.empty:
        errors.append("The uploaded CSV is empty.")
        return None, errors, warnings

    data = standardize_dataset(data)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    unexpected_columns = [column for column in data.columns if column not in required_columns]

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if unexpected_columns:
        warnings.append(f"Unexpected columns will be ignored: {', '.join(unexpected_columns)}")

    if TARGET_COLUMN in data.columns and data[TARGET_COLUMN].isna().any():
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


def render_algorithm_notes() -> None:
    st.markdown(
        """
        **Algorithm guide**

        - **Logistic Regression:** estimates purchase probability using a regularized linear decision boundary.
        - **Decision Tree:** splits visitor sessions into rule-based groups using impurity reduction.
        - **K-Nearest Neighbors:** predicts from nearby sessions, so scaled numeric features are important.
        - **Naive Bayes:** applies Bayes' theorem with a simplifying independence assumption.
        - **Support Vector Machine:** finds a maximum-margin boundary; this project uses an RBF kernel and probabilities for AUC.
        """
    )


def main() -> None:
    st.title("Online Shopper Purchase Intention Classifier")
    st.caption("BITS Pilani M.Tech AIML Assignment 2 | Saved-model evaluation, no retraining in the app")

    if not METADATA_FILE.exists():
        st.error("Project metadata is missing. Run `py -m src.train_models` from the project root first.")
        st.stop()

    metadata = load_metadata()
    models = load_models()

    with st.sidebar:
        st.header("Evaluation Controls")
        selected_model = st.selectbox(
            "Choose a classifier",
            list(MODEL_DISPLAY_NAMES.values()),
        )
        uploaded_file = st.file_uploader("Upload labeled test CSV", type=["csv"])
        st.divider()
        st.write("Expected target column:", f"`{TARGET_COLUMN}`")
        st.write("Expected feature count:", len(FEATURE_COLUMNS))

    tab_evaluate, tab_compare, tab_about = st.tabs(["Evaluate Upload", "Compare All Models", "Dataset & Algorithms"])

    if uploaded_file is None:
        st.info("Upload `test_data.csv` to evaluate the saved classifiers.")
        reference_metrics = load_reference_metrics()
        if not reference_metrics.empty:
            st.subheader("Reference Metrics from Held-Out Test Data")
            st.dataframe(reference_metrics, use_container_width=True, hide_index=True)
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

    validated_data, errors, warnings = validate_uploaded_data(uploaded_data, metadata)
    for warning in warnings:
        st.warning(warning)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    assert validated_data is not None
    x_uploaded = validated_data[FEATURE_COLUMNS]
    y_uploaded = validated_data[TARGET_COLUMN]

    with tab_evaluate:
        st.subheader("Uploaded Dataset")
        dim_col, class_col = st.columns([1, 2])
        dim_col.metric("Rows", validated_data.shape[0])
        dim_col.metric("Columns", validated_data.shape[1])
        class_col.bar_chart(y_uploaded.value_counts())
        st.dataframe(validated_data.head(20), use_container_width=True)

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
        st.caption("All scores are calculated from the uploaded labeled test CSV using saved pipelines only.")

    with tab_about:
        st.subheader("Dataset Context")
        st.write(
            "The dataset contains online shopping sessions from UCI. The target label `Revenue` indicates "
            "whether a session ended with a purchase."
        )
        st.write("Source:", metadata["dataset_source_url"])
        st.write("Training/test split:", metadata["split"])
        render_algorithm_notes()


if __name__ == "__main__":
    main()
