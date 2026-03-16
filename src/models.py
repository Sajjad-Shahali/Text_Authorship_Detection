"""
models.py
---------
Model registry and factory.

All classifiers are sklearn-compatible and work with sparse TF-IDF matrices.
Regularisation is applied by default to prevent overfitting.

Available models
----------------
Linear / margin-based (best for high-dim sparse text):
  logistic_regression         — LR, uniform class weights
  logistic_regression_balanced— LR, class_weight='balanced' (helps imbalanced data)
  linear_svc                  — LinearSVC, uniform weights
  linear_svc_balanced         — LinearSVC, class_weight='balanced'
  calibrated_svc              — LinearSVC wrapped in CalibratedClassifierCV (gives proba)
  sgd_log                     — SGDClassifier log-loss (fast LR equiv, balanced weights)
  sgd_hinge                   — SGDClassifier hinge-loss (fast SVM equiv, balanced weights)
  ridge_classifier            — RidgeClassifier (fast, good baseline)
  passive_aggressive          — PassiveAggressiveClassifier

Probabilistic / generative:
  complement_nb               — ComplementNB (good for imbalanced text classes)
"""

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.utils import get_logger

logger = get_logger(__name__)

AVAILABLE_MODELS = [
    "logistic_regression",
    "logistic_regression_balanced",
    "linear_svc",
    "linear_svc_balanced",
    "calibrated_svc",
    "sgd_log",
    "sgd_hinge",
    "ridge_classifier",
    "passive_aggressive",
    "complement_nb",
]


def get_model(name: str, config: Dict) -> BaseEstimator:
    """
    Factory — return an unfitted sklearn estimator by name.

    Parameters
    ----------
    name   : one of AVAILABLE_MODELS
    config : full project config dict (models section is used)

    Returns
    -------
    Unfitted sklearn estimator
    """
    model_cfg = config.get("models", {})
    seed = config.get("training", {}).get("random_state", 42)

    # ── Logistic Regression ──────────────────────────────────────────────────
    if name == "logistic_regression":
        cfg = model_cfg.get("logistic_regression", {})
        return LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=cfg.get("class_weight", None),
            random_state=seed,
        )

    if name == "logistic_regression_balanced":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        return LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )

    # ── LinearSVC ────────────────────────────────────────────────────────────
    if name == "linear_svc":
        cfg = model_cfg.get("linear_svc", {})
        return LinearSVC(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 2000),
            class_weight=cfg.get("class_weight", None),
            random_state=seed,
        )

    if name == "linear_svc_balanced":
        cfg = model_cfg.get("linear_svc_balanced", {})
        return LinearSVC(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 2000),
            class_weight="balanced",
            random_state=seed,
        )

    # ── CalibratedClassifierCV (LinearSVC + Platt scaling) ───────────────────
    if name == "calibrated_svc":
        cfg = model_cfg.get("calibrated_svc", {})
        base = LinearSVC(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 2000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=seed,
        )
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")

    # ── SGD Classifier ────────────────────────────────────────────────────────
    if name == "sgd_log":
        cfg = model_cfg.get("sgd_log", {})
        return SGDClassifier(
            loss="log_loss",
            alpha=cfg.get("alpha", 1e-4),
            max_iter=cfg.get("max_iter", 100),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    if name == "sgd_hinge":
        cfg = model_cfg.get("sgd_hinge", {})
        return SGDClassifier(
            loss="hinge",
            alpha=cfg.get("alpha", 1e-4),
            max_iter=cfg.get("max_iter", 100),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    # ── Ridge Classifier ──────────────────────────────────────────────────────
    if name == "ridge_classifier":
        cfg = model_cfg.get("ridge_classifier", {})
        return RidgeClassifier(
            alpha=cfg.get("alpha", 1.0),
            class_weight=cfg.get("class_weight", "balanced"),
        )

    # ── Passive Aggressive ────────────────────────────────────────────────────
    if name == "passive_aggressive":
        cfg = model_cfg.get("passive_aggressive", {})
        return PassiveAggressiveClassifier(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )

    # ── ComplementNB ──────────────────────────────────────────────────────────
    if name == "complement_nb":
        cfg = model_cfg.get("complement_nb", {})
        return ComplementNB(alpha=cfg.get("alpha", 0.1))

    raise ValueError(f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}")


def get_all_models(config: Dict) -> Dict[str, BaseEstimator]:
    """Return dict of all models listed in config.models.run_models."""
    run_models = config.get("models", {}).get("run_models", AVAILABLE_MODELS)
    return {name: get_model(name, config) for name in run_models}
