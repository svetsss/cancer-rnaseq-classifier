from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = 401
DATASET_NAME = "Gene Expression Cancer RNA-Seq"
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
    "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
)
DATASET_ARCHIVE_NAME = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
DATA_MEMBER = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
LABELS_MEMBER = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_ARCHIVE_PATH = RAW_DATA_DIR / DATASET_ARCHIVE_NAME
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_summary.json"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
SPLIT_SUMMARY_PATH = RESULTS_DIR / "split_summary.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
CLASS_DISTRIBUTION_PATH = FIGURES_DIR / "class_distribution.png"
