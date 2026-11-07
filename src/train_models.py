from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATASET_NAME,
    DATASET_SOURCE_URL,
    EVALUATION_RESULTS_FILE,
    FEATURE_COLUMNS,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_DISPLAY_NAMES,
    MODEL_FILENAMES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OBSERVATIONS_FILE,
    POSITIVE_CLASS,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RANDOM_SEED,
    RESULTS_DIR,
    ROOT_TEST_DATA_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_validation import download_raw_dataset, save_json, validate_assignment_requirements
from src.evaluate_models import evaluate_models
from src.preprocessing import build_preprocessor
from src.visualization import create_eda_visualizations, create_metrics_visualization


def ensure_directories() -> None:
    for path in [MODELS_DIR, RESULTS_DIR, ARTIFACTS_DIR, PROCESSED_TRAIN_FILE.parent]:
        path.mkdir(parents=True, exist_ok=True)


def build_model_pipelines() -> dict[str, Pipeline]:
    scaled_preprocessor = build_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        scale_numeric=True,
    )
    unscaled_preprocessor = build_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        scale_numeric=False,
    )

    return {
        MODEL_DISPLAY_NAMES["logistic_regression"]: Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        MODEL_DISPLAY_NAMES["decision_tree"]: Pipeline(
            steps=[
                ("preprocessor", unscaled_preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=20,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        MODEL_DISPLAY_NAMES["knn"]: Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=11, weights="distance")),
            ]
        ),
        MODEL_DISPLAY_NAMES["naive_bayes"]: Pipeline(
            steps=[
                ("preprocessor", unscaled_preprocessor),
                ("classifier", GaussianNB()),
            ]
        ),
        MODEL_DISPLAY_NAMES["svm"]: Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def save_models(models: dict[str, Pipeline]) -> None:
    filename_by_display = {MODEL_DISPLAY_NAMES[key]: filename for key, filename in MODEL_FILENAMES.items()}
    for display_name, model in models.items():
        joblib.dump(model, MODELS_DIR / filename_by_display[display_name])


def verify_saved_models(models: dict[str, Pipeline], x_test: pd.DataFrame) -> dict[str, bool]:
    filename_by_display = {MODEL_DISPLAY_NAMES[key]: filename for key, filename in MODEL_FILENAMES.items()}
    verification = {}
    for display_name, model in models.items():
        loaded = joblib.load(MODELS_DIR / filename_by_display[display_name])
        original_predictions = model.predict(x_test)
        loaded_predictions = loaded.predict(x_test)
        verification[display_name] = bool((original_predictions == loaded_predictions).all())
    return verification


def generate_observations(metrics: pd.DataFrame) -> str:
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    winners = {
        metric: metrics.sort_values(metric, ascending=False).iloc[0]["ML Model Name"]
        for metric in metric_columns
    }
    overall_scores = metrics.set_index("ML Model Name")[metric_columns].mean(axis=1)
    overall_winner = overall_scores.sort_values(ascending=False).index[0]

    lines = [
        "# Model Observations",
        "",
        "| ML Model Name | Observation about model performance |",
        "|---|---|",
    ]
    for _, row in metrics.iterrows():
        model_name = row["ML Model Name"]
        strengths = [metric for metric, winner in winners.items() if winner == model_name]
        strength_text = (
            f" It leads on {', '.join(strengths)}."
            if strengths
            else " It does not lead any individual metric, but its balance still matters."
        )
        observation = (
            f"Accuracy={row['Accuracy']:.4f}, AUC={row['AUC']:.4f}, "
            f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, "
            f"F1={row['F1']:.4f}, MCC={row['MCC']:.4f}."
            f"{strength_text} On this imbalanced purchase-intention dataset, AUC, F1, and MCC are "
            "especially important because accuracy alone can overstate majority-class performance."
        )
        lines.append(f"| {model_name} | {observation} |")

    lines.append(
        f"| Overall Winner for your dataset? | {overall_winner}. This is selected using the average rank across "
        "Accuracy, AUC, Precision, Recall, F1, and MCC rather than accuracy alone. |"
    )
    return "\n".join(lines) + "\n"


def write_metadata(
    data: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_verification: dict[str, bool],
) -> None:
    categorical_values = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_source_url": DATASET_SOURCE_URL,
        "target_column": TARGET_COLUMN,
        "input_feature_names": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_values_observed_in_training_dataset": categorical_values,
        "target_classes": sorted(data[TARGET_COLUMN].dropna().unique().tolist()),
        "positive_class": POSITIVE_CLASS,
        "split": {
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "stratified": True,
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "train_class_distribution": y_train.value_counts().astype(int).to_dict(),
            "test_class_distribution": y_test.value_counts().astype(int).to_dict(),
        },
        "metric_conventions": {
            "binary_positive_class": POSITIVE_CLASS,
            "auc": "roc_auc_score using predict_proba for TRUE class",
            "precision_recall_f1": "binary metrics using TRUE as positive class",
            "mcc": "sklearn.metrics.matthews_corrcoef",
        },
        "model_files": MODEL_FILENAMES,
        "training_configuration": {
            "logistic_regression": "LogisticRegression(max_iter=1000, solver='lbfgs') with scaled numeric features",
            "decision_tree": "DecisionTreeClassifier(max_depth=8, min_samples_leaf=20) with unscaled numeric features",
            "knn": "KNeighborsClassifier(n_neighbors=11, weights='distance') with scaled numeric features",
            "naive_bayes": "GaussianNB with dense one-hot categorical features and imputed numeric features",
            "svm": "SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced') with scaled numeric features",
        },
        "preprocessing_decisions": {
            "split_before_preprocessing_fit": True,
            "numeric_imputation": "median fitted on training data inside each pipeline",
            "categorical_imputation": "most_frequent fitted on training data inside each pipeline",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore') fitted on training data",
            "scaling": "StandardScaler only for Logistic Regression, KNN, and SVM numeric features",
            "identifier_columns_removed": [],
            "leakage_review": "No identifier column present. PageValues retained as documented analytics feature and discussed as strong predictor.",
        },
        "saved_model_prediction_verification": model_verification,
    }
    save_json(metadata, METADATA_FILE)


def main() -> None:
    ensure_directories()
    data = download_raw_dataset()

    validation_summary = validate_assignment_requirements(data)
    save_json(validation_summary, RESULTS_DIR / "data_validation_summary.json")
    create_eda_visualizations(data)

    x_values = data[FEATURE_COLUMNS]
    y_values = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_values,
    )

    train_export = x_train.copy()
    train_export[TARGET_COLUMN] = y_train
    test_export = x_test.copy()
    test_export[TARGET_COLUMN] = y_test
    train_export.to_csv(PROCESSED_TRAIN_FILE, index=False)
    test_export.to_csv(PROCESSED_TEST_FILE, index=False)
    test_export.to_csv(ROOT_TEST_DATA_FILE, index=False)

    models = build_model_pipelines()
    for model in models.values():
        model.fit(x_train, y_train)

    metrics, details = evaluate_models(models, x_test, y_test)
    metrics.to_csv(METRICS_FILE, index=False)
    EVALUATION_RESULTS_FILE.write_text(json.dumps(details, indent=2), encoding="utf-8")
    create_metrics_visualization(metrics)

    save_models(models)
    model_verification = verify_saved_models(models, x_test)
    write_metadata(data, x_train, x_test, y_train, y_test, model_verification)

    OBSERVATIONS_FILE.write_text(generate_observations(metrics), encoding="utf-8")

    print("Training and evaluation completed.")
    print(metrics.to_string(index=False))
    print(f"Saved test data: {ROOT_TEST_DATA_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
