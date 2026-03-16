"""
main_train.py
-------------
Entry point for the full training pipeline.

Usage:
    python main_train.py --config configs/config.yaml

Pipeline steps:
  1. Load config
  2. Set up experiment directory and logging
  3. Load and validate training data
  4. (Optional) Run cross-validation and model comparison
  5. Train final model on all data
  6. Save pipeline, metrics, and artifacts
"""

import argparse
import sys
import time
from pathlib import Path

from src.constants import BEST_MODEL_FILE, CONFIG_SNAPSHOT_FILE, MODEL_COMPARISON_FILE, RUN_LOG_FILE
from src.data import get_texts_and_labels, load_train
from src.evaluate import (
    generate_classification_report,
    generate_confusion_matrix,
    error_analysis,
)
from src.train import run_model_comparison, train_final_model
from src.utils import (
    ensure_dir,
    get_experiment_dir,
    get_logger,
    load_config,
    log_system_info,
    resolve_paths,
    save_config_snapshot,
    save_json,
    save_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MALTO Text Authorship Detection — Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file (default: configs/config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_start = time.time()

    # ── Load config ────────────────────────────────────────────────────────────
    config = load_config(args.config)
    config = resolve_paths(config)

    paths = config["paths"]
    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("models", {})

    # ── Set up experiment directory ────────────────────────────────────────────
    exp_dir = None
    if experiment_cfg.get("enabled", True):
        prefix = experiment_cfg.get("name_prefix", "run")
        exp_dir = get_experiment_dir(paths["experiments_dir"], prefix=prefix)

    # Log file: inside experiment dir if available, else logs_dir
    log_file = str(exp_dir / RUN_LOG_FILE) if exp_dir else str(
        Path(paths.get("logs_dir", "artifacts/logs")) / "run.log"
    )

    logger = get_logger("main_train", log_file=log_file)
    log_system_info(logger)
    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"Experiment dir: {exp_dir}")

    # Save config snapshot for reproducibility
    if exp_dir:
        save_config_snapshot(config, exp_dir / CONFIG_SNAPSHOT_FILE)

    # ── Ensure artifact directories exist ─────────────────────────────────────
    for dir_key in ["models_dir", "metrics_dir", "submissions_dir", "analysis_dir"]:
        ensure_dir(paths[dir_key])

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading training data...")
    train_df = load_train(paths["train_file"])
    X, y = get_texts_and_labels(train_df)
    logger.info(f"Dataset: {len(X)} samples, {len(set(y))} classes")

    # ── Model comparison (CV) ──────────────────────────────────────────────────
    best_model_name = model_cfg.get("best_model", "logistic_regression")

    if training_cfg.get("run_cv", True) and training_cfg.get("run_model_comparison", True):
        logger.info("\nStarting model comparison via cross-validation...")
        comparison_df, cv_best_name, all_cv_results = run_model_comparison(
            X, y, config, experiment_dir=exp_dir
        )

        # Save model comparison
        comp_path = paths["metrics_dir"] + "/" + MODEL_COMPARISON_FILE
        comparison_df.to_csv(comp_path, index=False)
        logger.info(f"Model comparison saved to: {comp_path}")

        if exp_dir:
            comparison_df.to_csv(exp_dir / MODEL_COMPARISON_FILE, index=False)

        # Use CV winner if config says so
        # (config.models.best_model can override the CV winner for explicit control)
        logger.info(f"CV best model: {cv_best_name} | Config best model: {best_model_name}")

    elif training_cfg.get("run_cv", True):
        # Run CV for the single best model only
        from src.train import run_cross_validation
        logger.info(f"\nRunning CV for model: {best_model_name}")
        cv_results = run_cross_validation(X, y, best_model_name, config, experiment_dir=exp_dir)

        # Save metrics
        cv_path = paths["metrics_dir"] + "/cv_results.json"
        save_json(cv_results, cv_path)
        logger.info(f"CV results saved to: {cv_path}")

        report = cv_results["oof_classification_report"]
        report_path = paths["metrics_dir"] + "/classification_report.txt"
        save_text(report, report_path)

    # ── Final training ─────────────────────────────────────────────────────────
    model_save_path = paths["best_model_file"]
    logger.info(f"\nTraining final model: {best_model_name}")
    pipeline = train_final_model(X, y, best_model_name, config, save_path=model_save_path)

    if exp_dir:
        import joblib
        joblib.dump(pipeline, exp_dir / BEST_MODEL_FILE)
        logger.info(f"Final model also saved to experiment dir: {exp_dir / BEST_MODEL_FILE}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = time.time() - run_start
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Total time       : {total_time:.1f}s")
    logger.info(f"  Best model       : {best_model_name}")
    logger.info(f"  Model artifact   : {model_save_path}")
    if exp_dir:
        logger.info(f"  Experiment dir   : {exp_dir}")
    logger.info("=" * 60)
    logger.info("Next step: python main_infer.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
