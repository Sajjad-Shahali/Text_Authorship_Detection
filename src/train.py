"""
train.py
--------
Training logic: cross-validation loop, model comparison, and final fit.

Anti-leakage guarantee:
  The full feature pipeline (Preprocessor + FeatureUnion TF-IDF) is rebuilt
  and fitted INSIDE each CV fold. No vectorizer state bleeds across folds.

Outputs:
  - CV metrics per fold
  - OOF (out-of-fold) predictions for full-dataset error analysis
  - Model comparison table
  - Final trained pipeline (on all training data)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.constants import RANDOM_SEED
from src.evaluate import (
    compute_macro_f1,
    compute_metrics,
    error_analysis,
    generate_classification_report,
    generate_confusion_matrix,
    log_fold_metrics,
    summarise_cv_results,
)
from src.features import build_feature_union
from src.models import get_all_models, get_model
from src.preprocess import Preprocessor
from src.utils import ensure_dir, get_logger, save_json, save_text

logger = get_logger(__name__)


# ── Pipeline factory ──────────────────────────────────────────────────────────

def build_pipeline(model_name: str, config: Dict) -> Pipeline:
    """
    Build a full sklearn Pipeline:
      Preprocessor → FeatureUnion(TF-IDF) → Classifier

    The pipeline is unfitted. Fitting it inside a CV fold guarantees
    no leakage from val into train features.
    """
    preprocessor = Preprocessor.from_config(config)
    feature_union = build_feature_union(config)
    classifier = get_model(model_name, config)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("features", feature_union),
        ("classifier", classifier),
    ])
    return pipeline


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(
    X: List[str],
    y: np.ndarray,
    model_name: str,
    config: Dict,
    experiment_dir: Optional[Path] = None,
) -> Dict:
    """
    Run StratifiedKFold CV for a single model.

    Returns a results dict with per-fold and aggregate metrics,
    plus OOF predictions.
    """
    val_cfg = config.get("validation", {})
    n_splits = val_cfg.get("n_splits", 5)
    random_state = val_cfg.get("random_state", RANDOM_SEED)
    shuffle = val_cfg.get("shuffle", True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    X_arr = np.array(X, dtype=object)
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = None  # filled if model supports predict_proba

    fold_metrics = []

    logger.info(f"  Running {n_splits}-fold CV for model: {model_name}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y)):
        fold_start = time.time()

        X_train_fold = X_arr[train_idx].tolist()
        X_val_fold = X_arr[val_idx].tolist()
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Build and fit a FRESH pipeline — critical anti-leakage step
        pipeline = build_pipeline(model_name, config)
        pipeline.fit(X_train_fold, y_train_fold)

        # Train metrics (overfitting diagnostic)
        train_preds = pipeline.predict(X_train_fold)
        train_f1 = compute_macro_f1(y_train_fold, train_preds)

        # Validation metrics
        val_preds = pipeline.predict(X_val_fold)
        val_f1 = compute_macro_f1(y_val_fold, val_preds)

        oof_preds[val_idx] = val_preds

        # Collect probabilities if available
        if hasattr(pipeline, "predict_proba"):
            try:
                fold_proba = pipeline.predict_proba(X_val_fold)
                if oof_proba is None:
                    oof_proba = np.zeros((len(y), fold_proba.shape[1]))
                oof_proba[val_idx] = fold_proba
            except Exception:
                pass

        fold_time = time.time() - fold_start
        log_fold_metrics(fold, train_f1, val_f1, logger)
        logger.info(f"    Fold time: {fold_time:.1f}s")

        fold_metrics.append({
            "fold": fold + 1,
            "train_macro_f1": round(float(train_f1), 4),
            "val_macro_f1": round(float(val_f1), 4),
        })

    # Aggregate
    summary = summarise_cv_results(fold_metrics)
    logger.info(
        f"  {model_name} CV → "
        f"mean_val_f1={summary['mean_val_macro_f1']:.4f} ± {summary['std_val_macro_f1']:.4f}"
    )

    # OOF classification report and confusion matrix
    oof_report = generate_classification_report(y, oof_preds)
    oof_cm = generate_confusion_matrix(y, oof_preds)
    oof_metrics = compute_metrics(y, oof_preds)

    results = {
        "model_name": model_name,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "oof_metrics": oof_metrics,
        "oof_classification_report": oof_report,
    }

    # Save per-model artifacts if experiment_dir given
    if experiment_dir is not None:
        model_dir = experiment_dir / model_name
        ensure_dir(model_dir)
        save_json(results, model_dir / "cv_results.json")
        save_text(oof_report, model_dir / "classification_report.txt")
        oof_cm.to_csv(model_dir / "confusion_matrix.csv")
        logger.info(f"  Saved CV artifacts to: {model_dir}")

        # Error analysis
        if oof_proba is not None:
            top_n = config.get("analysis", {}).get("top_n_errors", 50)
            err_df = error_analysis(X, y, oof_preds, oof_proba, top_n=top_n)
            err_df.to_csv(model_dir / "error_analysis.csv", index=False)

    return results


# ── Model comparison ──────────────────────────────────────────────────────────

def run_model_comparison(
    X: List[str],
    y: np.ndarray,
    config: Dict,
    experiment_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Run CV for all configured models and return a comparison DataFrame
    along with the name of the best model (highest mean_val_macro_f1).
    """
    model_cfg = config.get("models", {})
    run_models = model_cfg.get("run_models", ["logistic_regression"])

    rows = []
    all_results = {}

    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    for model_name in run_models:
        logger.info(f"\n[Model: {model_name}]")
        results = run_cross_validation(X, y, model_name, config, experiment_dir)
        all_results[model_name] = results
        summary = results["summary"]
        oof_metrics = results["oof_metrics"]

        rows.append({
            "model": model_name,
            "mean_val_macro_f1": summary["mean_val_macro_f1"],
            "std_val_macro_f1": summary["std_val_macro_f1"],
            "mean_train_macro_f1": summary["mean_train_macro_f1"],
            "mean_overfit_gap": summary["mean_overfit_gap"],
            "oof_macro_f1": oof_metrics["macro_f1"],
            "oof_accuracy": oof_metrics["accuracy"],
        })

    comparison_df = pd.DataFrame(rows).sort_values(
        "mean_val_macro_f1", ascending=False
    ).reset_index(drop=True)

    logger.info("\nModel Comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))

    # Determine best model
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model"]
    logger.info(f"\nBest model: {best_model_name}  (val_f1={best_row['mean_val_macro_f1']:.4f})")

    return comparison_df, best_model_name, all_results


# ── Final training ────────────────────────────────────────────────────────────

def train_final_model(
    X: List[str],
    y: np.ndarray,
    model_name: str,
    config: Dict,
    save_path: Optional[str] = None,
) -> Pipeline:
    """
    Fit a full pipeline on ALL training data.
    This is the model used for test predictions.
    """
    logger.info("=" * 60)
    logger.info(f"FINAL TRAINING — model: {model_name}")
    logger.info(f"  Training samples: {len(X)}")

    start = time.time()
    pipeline = build_pipeline(model_name, config)
    pipeline.fit(X, y)
    elapsed = time.time() - start

    logger.info(f"  Final training completed in {elapsed:.1f}s")

    if save_path is not None:
        ensure_dir(Path(save_path).parent)
        joblib.dump(pipeline, save_path)
        logger.info(f"  Pipeline saved to: {save_path}")

    return pipeline
