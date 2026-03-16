"""
data.py
-------
Safe data loading with schema validation.
Raises descriptive errors on malformed input rather than silently proceeding.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.constants import (
    MAX_LABEL,
    MIN_LABEL,
    SUBMISSION_ID_COL,
    SUBMISSION_LABEL_COL,
    TEST_TEXT_COL,
    TRAIN_LABEL_COL,
    TRAIN_TEXT_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ── Train ─────────────────────────────────────────────────────────────────────

def load_train(path: str) -> pd.DataFrame:
    """
    Load training CSV and validate schema.

    Expected columns: `text`, `label`
    Returns validated DataFrame.
    """
    path = Path(path)
    logger.info(f"Loading train data from: {path}")
    _assert_file_exists(path)

    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")

    _validate_train_schema(df)
    df = _clean_train(df)

    logger.info(f"  Cleaned shape: {df.shape}")
    logger.info(f"  Class distribution:\n{df[TRAIN_LABEL_COL].value_counts().sort_index().to_string()}")
    return df


def _validate_train_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing or data is malformed."""
    required = {TRAIN_TEXT_COL, TRAIN_LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Train CSV missing required columns: {missing}")

    # Label range check
    bad_labels = df[TRAIN_LABEL_COL].dropna()
    bad_labels = bad_labels[~bad_labels.isin(range(MIN_LABEL, MAX_LABEL + 1))]
    if not bad_labels.empty:
        raise ValueError(
            f"Train labels out of range [{MIN_LABEL}, {MAX_LABEL}]: "
            f"{bad_labels.unique().tolist()[:10]}"
        )

    # Fully empty text column
    if df[TRAIN_TEXT_COL].isna().all():
        raise ValueError("Train CSV: text column is entirely empty.")


def _clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null text or null label; coerce label to int."""
    before = len(df)
    df = df.dropna(subset=[TRAIN_TEXT_COL, TRAIN_LABEL_COL]).copy()
    df[TRAIN_LABEL_COL] = df[TRAIN_LABEL_COL].astype(int)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"  Dropped {dropped} rows with null text or label.")
    return df.reset_index(drop=True)


# ── Test ──────────────────────────────────────────────────────────────────────

def load_test(path: str) -> pd.DataFrame:
    """
    Load test CSV and validate schema.

    Expected columns: `text`
    Returns validated DataFrame.
    """
    path = Path(path)
    logger.info(f"Loading test data from: {path}")
    _assert_file_exists(path)

    df = pd.read_csv(path, encoding="utf-8")
    # Drop unnamed index columns that some export tools add (e.g. 'Unnamed: 0')
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        logger.info(f"  Dropped unnamed index columns: {unnamed}")
    logger.info(f"  Raw shape: {df.shape}")

    _validate_test_schema(df)
    df = _clean_test(df)

    logger.info(f"  Cleaned shape: {df.shape}")
    return df


def _validate_test_schema(df: pd.DataFrame) -> None:
    if TEST_TEXT_COL not in df.columns:
        raise ValueError(f"Test CSV missing required column: '{TEST_TEXT_COL}'")
    if df[TEST_TEXT_COL].isna().all():
        raise ValueError("Test CSV: text column is entirely empty.")


def _clean_test(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[TEST_TEXT_COL]).copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"  Test: dropped {dropped} rows with null text.")
    return df.reset_index(drop=True)


# ── Sample submission ─────────────────────────────────────────────────────────

def load_sample_submission(path: str) -> pd.DataFrame:
    """
    Load sample_submission.csv for format validation.
    Returns None (with a warning) if the file does not exist,
    so callers can skip row-count validation gracefully.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"sample_submission.csv not found at {path} — skipping row count check.")
        return None

    logger.info(f"Loading sample submission from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    required = {SUBMISSION_ID_COL, SUBMISSION_LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample submission missing columns: {missing}")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")


def get_texts_and_labels(df: pd.DataFrame) -> Tuple:
    """Return (X, y) arrays from a loaded train DataFrame."""
    X = df[TRAIN_TEXT_COL].tolist()
    y = df[TRAIN_LABEL_COL].values
    return X, y


def get_test_texts(df: pd.DataFrame):
    """Return list of text strings from a loaded test DataFrame."""
    return df[TEST_TEXT_COL].tolist()
