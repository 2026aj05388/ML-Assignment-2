from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import POSITIVE_CLASS


METRIC_COLUMNS = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _probability_scores(model: Any, x_values: pd.DataFrame, class_labels: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_values)
        if len(class_labels) == 2:
            positive_index = list(model.classes_).index(POSITIVE_CLASS)
            return probabilities[:, positive_index]
        return probabilities

    if hasattr(model, "decision_function"):
        return model.decision_function(x_values)

    raise ValueError("AUC requires predict_proba or decision_function, but neither is available.")


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str,
) -> tuple[dict[str, float | str], dict[str, Any]]:
    labels = list(pd.Series(y_test).dropna().unique())
    labels = sorted(labels)

    predictions = model.predict(x_test)
    scores = _probability_scores(model, x_test, labels)

    if len(labels) == 2:
        auc = roc_auc_score((y_test == POSITIVE_CLASS).astype(int), scores)
        precision = precision_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        recall = recall_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        averaging = "binary, positive class TRUE"
    else:
        auc = roc_auc_score(y_test, scores, multi_class="ovr", average="macro", labels=labels)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        averaging = "multiclass weighted metrics, macro OVR AUC"

    metric_row = {
        "ML Model Name": model_name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "AUC": round(float(auc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(matthews_corrcoef(y_test, predictions)), 4),
    }

    detailed_results = {
        "model_name": model_name,
        "averaging": averaging,
        "class_labels": labels,
        "predictions": predictions.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }
    return metric_row, detailed_results


def evaluate_models(
    models: dict[str, Any],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    details = {}
    for model_name, model in models.items():
        row, model_details = evaluate_model(model, x_test, y_test, model_name=model_name)
        rows.append(row)
        details[model_name] = model_details

    return pd.DataFrame(rows, columns=METRIC_COLUMNS), details
