"""
constants.py
------------
Project-wide constants. Single source of truth for label maps,
column names, and fixed values.
"""

# ── Label definitions ─────────────────────────────────────────────────────────

LABEL_MAP = {
    0: "Human",
    1: "DeepSeek",
    2: "Grok",
    3: "Claude",
    4: "Gemini",
    5: "ChatGPT",
}

LABEL_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]

NUM_CLASSES = len(LABEL_MAP)
MIN_LABEL = 0
MAX_LABEL = 5

# ── Column names ──────────────────────────────────────────────────────────────

TRAIN_TEXT_COL = "TEXT"
TRAIN_LABEL_COL = "LABEL"

TEST_TEXT_COL = "TEXT"

SUBMISSION_ID_COL = "ID"
SUBMISSION_LABEL_COL = "LABEL"

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ── Artifact filenames ────────────────────────────────────────────────────────

CV_RESULTS_FILE = "cv_results.json"
MODEL_COMPARISON_FILE = "model_comparison.csv"
CLASSIFICATION_REPORT_FILE = "classification_report.txt"
CONFUSION_MATRIX_FILE = "confusion_matrix.csv"
ERROR_ANALYSIS_FILE = "error_analysis.csv"
BEST_MODEL_FILE = "best_model.joblib"
CONFIG_SNAPSHOT_FILE = "config_snapshot.yaml"
RUN_LOG_FILE = "run.log"
SUBMISSION_FILE = "submission.csv"
SUBMISSION_LATEST_FILE = "submission_latest.csv"
