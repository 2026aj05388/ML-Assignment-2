from pathlib import Path


RANDOM_SEED = 42
TEST_SIZE = 0.20

DATASET_NAME = "Online Shoppers Purchasing Intention Dataset"
DATASET_SOURCE_URL = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
DATASET_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILE = RAW_DATA_DIR / "original_dataset.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_data.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "heldout_test_data.csv"
ROOT_TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"
METRICS_FILE = RESULTS_DIR / "metrics.csv"
EVALUATION_RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"
OBSERVATIONS_FILE = RESULTS_DIR / "model_observations.md"
METADATA_FILE = ARTIFACTS_DIR / "metadata.json"

TARGET_COLUMN = "Revenue"
POSITIVE_CLASS = "TRUE"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}

MODEL_FILENAMES = {
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree": "decision_tree.joblib",
    "knn": "knn.joblib",
    "naive_bayes": "naive_bayes.joblib",
    "svm": "svm.joblib",
}
