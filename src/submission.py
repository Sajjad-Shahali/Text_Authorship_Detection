"""
submission.py
-------------
Format, validate, and save Kaggle submission files.

Expected output format:
  ID,LABEL
  0,5
  1,0
  2,3
  ...

Validation is performed against sample_submission.csv to catch
row count mismatches and column name errors before submission.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.constants import MAX_LABEL, MIN_LABEL, SUBMISSION_ID_COL, SUBMISSION_LABEL_COL
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def make_submission(predictions: np.ndarray) -> pd.DataFrame:
    """
    Wrap predictions in a submission DataFrame.

    Parameters
    ----------
    predictions : np.ndarray of int
        Model predictions, one per test sample.

    Returns
    -------
    pd.DataFrame with columns [ID, LABEL]
    """
    ids = np.arange(len(predictions), dtype=int)
    df = pd.DataFrame({
        SUBMISSION_ID_COL: ids,
        SUBMISSION_LABEL_COL: predictions.astype(int),
    })
    logger.info(f"Created submission with {len(df)} rows.")
    return df


def validate_submission(
    submission_df: pd.DataFrame,
    sample_df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Validate submission format.

    Checks:
    - Required columns present
    - LABEL values are integers in [0, 5]
    - No null values
    - Row count matches sample_submission (if provided)
    - ID is zero-indexed sequential

    Returns True if valid, raises ValueError on failure.
    """
    errors = []

    # Column check
    required_cols = {SUBMISSION_ID_COL, SUBMISSION_LABEL_COL}
    missing = required_cols - set(submission_df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        raise ValueError(f"Submission validation failed: {errors}")

    # Null check
    if submission_df[SUBMISSION_LABEL_COL].isna().any():
        errors.append("LABEL column contains null values.")

    # Label range check
    bad = submission_df[SUBMISSION_LABEL_COL]
    bad = bad[(bad < MIN_LABEL) | (bad > MAX_LABEL)]
    if not bad.empty:
        errors.append(
            f"LABEL values out of range [{MIN_LABEL}, {MAX_LABEL}]: "
            f"{bad.unique().tolist()}"
        )

    # Row count match
    if sample_df is not None:
        if len(submission_df) != len(sample_df):
            errors.append(
                f"Row count mismatch: submission has {len(submission_df)} rows, "
                f"sample has {len(sample_df)} rows."
            )

    # ID check: should be 0-indexed sequential
    expected_ids = list(range(len(submission_df)))
    actual_ids = submission_df[SUBMISSION_ID_COL].tolist()
    if actual_ids != expected_ids:
        errors.append(
            f"ID column is not sequential 0-indexed. "
            f"First 5 expected={expected_ids[:5]}, actual={actual_ids[:5]}"
        )

    if errors:
        raise ValueError(f"Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    logger.info("Submission validation passed.")
    return True


def save_submission(
    df: pd.DataFrame,
    path: str,
    also_save_latest: bool = True,
) -> None:
    """
    Save submission DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Submission DataFrame with columns [ID, LABEL]
    path : str
        Output path for the timestamped submission file.
    also_save_latest : bool
        If True, also save a copy as submission_latest.csv in the same directory.
    """
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)
    logger.info(f"Submission saved to: {path}")

    if also_save_latest:
        latest_path = Path(path).parent / "submission_latest.csv"
        df.to_csv(latest_path, index=False)
        logger.info(f"Latest submission also saved to: {latest_path}")
