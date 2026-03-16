"""
main_submit.py
--------------
Entry point for generating and validating the Kaggle submission file.

Usage:
    python main_submit.py --config configs/config.yaml
    python main_submit.py --config configs/config.yaml --predictions artifacts/submissions/submission_latest.csv

This script can either:
  (a) Run inference + generate submission in one step (default)
  (b) Validate and reformat an existing predictions CSV

Run (a) is equivalent to: python main_infer.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import load_sample_submission, load_test, get_test_texts
from src.inference import load_pipeline, predict
from src.submission import make_submission, save_submission, validate_submission
from src.utils import (
    ensure_dir,
    get_logger,
    load_config,
    log_system_info,
    resolve_paths,
)
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="MALTO Text Authorship Detection — Submission Generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help=(
            "Path to an existing predictions CSV to validate and reformat. "
            "If not provided, inference will be run from scratch."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for submission CSV (optional override)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    config = resolve_paths(config)
    paths = config["paths"]

    logger = get_logger("main_submit")
    log_system_info(logger)
    logger.info(f"Config loaded from: {args.config}")

    ensure_dir(paths["submissions_dir"])

    import numpy as np

    # ── Option A: load existing predictions ───────────────────────────────────
    if args.predictions:
        logger.info(f"Loading predictions from: {args.predictions}")
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            logger.error(f"Predictions file not found: {pred_path}")
            sys.exit(1)

        pred_df = pd.read_csv(pred_path)
        from src.constants import SUBMISSION_LABEL_COL, SUBMISSION_ID_COL
        if SUBMISSION_LABEL_COL in pred_df.columns:
            preds = pred_df[SUBMISSION_LABEL_COL].values.astype(int)
        elif "label" in pred_df.columns:
            preds = pred_df["label"].values.astype(int)
        else:
            logger.error(f"Cannot find label column in {args.predictions}. "
                         f"Columns: {pred_df.columns.tolist()}")
            sys.exit(1)

    # ── Option B: run inference from scratch ──────────────────────────────────
    else:
        logger.info("No --predictions provided. Running inference...")
        pipeline = load_pipeline(paths["best_model_file"])
        test_df = load_test(paths["test_file"])
        X_test = get_test_texts(test_df)
        preds = predict(pipeline, X_test)

    # ── Build and validate submission ─────────────────────────────────────────
    submission_df = make_submission(preds)

    sample_path = paths.get("sample_submission_file")
    sample_df = load_sample_submission(sample_path) if sample_path else None
    validate_submission(submission_df, sample_df)

    # ── Save ───────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = args.output
    else:
        out_path = str(Path(paths["submissions_dir"]) / f"submission_{timestamp}.csv")

    save_submission(submission_df, out_path, also_save_latest=True)

    logger.info("=" * 60)
    logger.info("SUBMISSION READY")
    logger.info(f"  Rows     : {len(submission_df)}")
    logger.info(f"  File     : {out_path}")
    logger.info("=" * 60)
    logger.info("Upload to Kaggle: 'Submit Predictions' on the competition page.")


if __name__ == "__main__":
    main()
