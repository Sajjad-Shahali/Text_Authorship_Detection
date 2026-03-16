"""
test_submission_format.py
-------------------------
Tests for submission format validation.
Verifies correct column names, label ranges, row counts, and ID format.
"""

import numpy as np
import pandas as pd
import pytest

from src.submission import make_submission, validate_submission, save_submission
from src.constants import SUBMISSION_ID_COL, SUBMISSION_LABEL_COL, MIN_LABEL, MAX_LABEL


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_sample_submission(n=10) -> pd.DataFrame:
    """Return a synthetic sample_submission DataFrame."""
    return pd.DataFrame({
        SUBMISSION_ID_COL: list(range(n)),
        SUBMISSION_LABEL_COL: [0] * n,
    })


def make_valid_predictions(n=10) -> np.ndarray:
    """Return valid integer predictions cycling through [0, 5]."""
    return np.array([i % 6 for i in range(n)], dtype=int)


# ── make_submission tests ─────────────────────────────────────────────────────

def test_make_submission_returns_dataframe():
    """make_submission should return a DataFrame."""
    preds = make_valid_predictions(10)
    df = make_submission(preds)
    assert isinstance(df, pd.DataFrame)


def test_make_submission_columns():
    """Submission DataFrame must have ID and LABEL columns."""
    preds = make_valid_predictions(5)
    df = make_submission(preds)
    assert SUBMISSION_ID_COL in df.columns
    assert SUBMISSION_LABEL_COL in df.columns


def test_make_submission_row_count():
    """Row count must match number of predictions."""
    n = 15
    preds = make_valid_predictions(n)
    df = make_submission(preds)
    assert len(df) == n


def test_make_submission_id_is_zero_indexed():
    """ID column must be 0, 1, 2, ..., n-1."""
    n = 8
    preds = make_valid_predictions(n)
    df = make_submission(preds)
    assert df[SUBMISSION_ID_COL].tolist() == list(range(n))


def test_make_submission_labels_are_integers():
    """LABEL column values must be integers."""
    preds = make_valid_predictions(6)
    df = make_submission(preds)
    assert df[SUBMISSION_LABEL_COL].dtype in [int, np.int32, np.int64]


# ── validate_submission tests ─────────────────────────────────────────────────

def test_validate_passes_on_valid_submission():
    """Valid submission should pass validation without errors."""
    n = 10
    preds = make_valid_predictions(n)
    submission_df = make_submission(preds)
    sample_df = make_sample_submission(n)
    result = validate_submission(submission_df, sample_df)
    assert result is True


def test_validate_fails_on_missing_label_column():
    """Missing LABEL column should raise ValueError."""
    df = pd.DataFrame({SUBMISSION_ID_COL: [0, 1, 2]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_submission(df)


def test_validate_fails_on_missing_id_column():
    """Missing ID column should raise ValueError."""
    df = pd.DataFrame({SUBMISSION_LABEL_COL: [0, 1, 2]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_submission(df)


def test_validate_fails_on_out_of_range_labels():
    """Labels outside [0, 5] should fail validation."""
    df = pd.DataFrame({
        SUBMISSION_ID_COL: [0, 1, 2],
        SUBMISSION_LABEL_COL: [0, 99, 3],  # 99 is invalid
    })
    with pytest.raises(ValueError, match="out of range"):
        validate_submission(df)


def test_validate_fails_on_row_count_mismatch():
    """Row count mismatch between submission and sample should raise ValueError."""
    preds = make_valid_predictions(10)
    submission_df = make_submission(preds)
    sample_df = make_sample_submission(5)  # different size

    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_submission(submission_df, sample_df)


def test_validate_fails_on_null_labels():
    """Null values in LABEL column should fail validation."""
    df = pd.DataFrame({
        SUBMISSION_ID_COL: [0, 1, 2],
        SUBMISSION_LABEL_COL: [0, None, 3],
    })
    with pytest.raises(ValueError):
        validate_submission(df)


def test_validate_all_valid_labels():
    """Each valid label value [0-5] individually should pass."""
    for label in range(MIN_LABEL, MAX_LABEL + 1):
        df = pd.DataFrame({
            SUBMISSION_ID_COL: [0],
            SUBMISSION_LABEL_COL: [label],
        })
        result = validate_submission(df)
        assert result is True


# ── save_submission tests ─────────────────────────────────────────────────────

def test_save_submission_creates_file(tmp_path):
    """save_submission should write a CSV file to disk."""
    preds = make_valid_predictions(5)
    df = make_submission(preds)
    out_path = str(tmp_path / "submission.csv")

    save_submission(df, out_path, also_save_latest=False)

    import os
    assert os.path.exists(out_path)


def test_save_submission_readable(tmp_path):
    """Saved submission should be readable as a CSV with correct content."""
    preds = make_valid_predictions(5)
    df = make_submission(preds)
    out_path = str(tmp_path / "submission.csv")

    save_submission(df, out_path, also_save_latest=False)
    loaded = pd.read_csv(out_path)

    assert SUBMISSION_ID_COL in loaded.columns
    assert SUBMISSION_LABEL_COL in loaded.columns
    assert len(loaded) == 5


def test_save_submission_also_saves_latest(tmp_path):
    """also_save_latest=True should create submission_latest.csv."""
    preds = make_valid_predictions(5)
    df = make_submission(preds)
    out_path = str(tmp_path / "submission_20260316.csv")

    save_submission(df, out_path, also_save_latest=True)

    import os
    latest_path = str(tmp_path / "submission_latest.csv")
    assert os.path.exists(latest_path)
