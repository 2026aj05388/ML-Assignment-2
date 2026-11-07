from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, NUMERIC_FEATURES, TARGET_COLUMN


def _save_current_plot(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def create_eda_visualizations(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=data, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2", legend=False)
    plt.title("Target Class Distribution: Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Session Count")
    _save_current_plot("target_class_distribution.png")

    missing_counts = data.isna().sum().to_frame("missing_count")
    plt.figure(figsize=(8, 4))
    sns.heatmap(missing_counts.T, cmap="Reds", cbar=False)
    plt.title("Missing Value Check by Column")
    plt.yticks(rotation=0)
    _save_current_plot("missing_value_heatmap.png")

    sample_numeric = NUMERIC_FEATURES[:6]
    data[sample_numeric].hist(figsize=(12, 8), bins=30)
    plt.suptitle("Numerical Feature Distributions", y=1.02)
    _save_current_plot("numeric_feature_distributions.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET_COLUMN, y="PageValues", hue=TARGET_COLUMN, palette="Set2", legend=False)
    plt.title("PageValues by Revenue Class")
    plt.xlabel("Revenue")
    plt.ylabel("PageValues")
    _save_current_plot("page_values_by_revenue.png")

    plt.figure(figsize=(10, 8))
    corr = data[NUMERIC_FEATURES].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap for Numerical Features")
    _save_current_plot("numeric_correlation_heatmap.png")


def create_metrics_visualization(metrics: pd.DataFrame) -> None:
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    long_metrics = metrics.melt(
        id_vars="ML Model Name",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=long_metrics, x="Metric", y="Score", hue="ML Model Name")
    plt.title("Model Performance Comparison on Held-Out Test Data")
    plt.ylim(0, 1)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    _save_current_plot("model_metrics_comparison.png")
