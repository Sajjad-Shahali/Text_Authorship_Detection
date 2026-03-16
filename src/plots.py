"""
plots.py
--------
All training visualisations:

  1. Learning curve per model  — training score & CV score vs. training set size
  2. Overfitting plot per model — train vs. val Macro F1 across CV folds
  3. Model comparison bar chart — all models ranked by CV Macro F1
  4. Confusion matrix heatmap   — OOF confusion matrix

Plots are saved to artifacts/plots/ (or a run-specific subdirectory).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts with no display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# Shared style
PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]


# ── 1. Learning curve ─────────────────────────────────────────────────────────

def compute_learning_curve(pipeline, X, y, config: Dict):
    """
    Compute learning curve data using sklearn's learning_curve.
    Returns (train_sizes_abs, train_scores_mean, train_scores_std,
             val_scores_mean, val_scores_std).
    """
    from sklearn.model_selection import learning_curve, StratifiedKFold

    val_cfg = config.get("validation", {})
    # Use fewer folds (3) for speed; learning curves are diagnostic, not final metrics
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=val_cfg.get("random_state", 42),
    )

    lc_cfg = config.get("learning_curve", {})
    n_points = lc_cfg.get("n_points", 6)
    train_sizes = np.linspace(0.15, 1.0, n_points)

    logger.info(f"  Computing learning curve ({n_points} points, 3-fold CV)...")

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        train_sizes=train_sizes,
        n_jobs=1,       # keep deterministic; parallelism here causes issues on Windows
        error_score="raise",
    )

    return (
        train_sizes_abs,
        train_scores.mean(axis=1),
        train_scores.std(axis=1),
        val_scores.mean(axis=1),
        val_scores.std(axis=1),
    )


def plot_learning_curve(
    model_name: str,
    train_sizes: np.ndarray,
    train_scores_mean: np.ndarray,
    train_scores_std: np.ndarray,
    val_scores_mean: np.ndarray,
    val_scores_std: np.ndarray,
    save_path: str,
):
    """
    Plot and save a learning curve for one model.

    Shows training score and cross-validation score vs. training set size.
    The gap between curves reveals overfitting.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Training score band
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.12, color=PALETTE[0],
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color=PALETTE[0],
            linewidth=2, markersize=5, label="Training Macro F1")

    # Validation score band
    ax.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.12, color=PALETTE[1],
    )
    ax.plot(train_sizes, val_scores_mean, "s-", color=PALETTE[1],
            linewidth=2, markersize=5, label="CV Validation Macro F1")

    # Annotations
    final_gap = float(train_scores_mean[-1] - val_scores_mean[-1])
    ax.set_title(
        f"Learning Curve — {model_name}\n"
        f"Final gap (train − val): {final_gap:+.4f}",
        fontsize=12,
    )
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Macro F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved learning curve: {save_path}")


def plot_all_learning_curves(
    lc_results: Dict[str, dict],
    save_path: str,
):
    """
    Single figure with all models' learning curves in a grid.

    lc_results: {model_name: {train_sizes, train_mean, train_std, val_mean, val_std}}
    """
    n = len(lc_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, (model_name, lc) in enumerate(lc_results.items()):
        ax = axes_flat[idx]
        ts     = np.array(lc["train_sizes"])
        tm     = np.array(lc["train_mean"])
        ts_std = np.array(lc["train_std"])
        vm     = np.array(lc["val_mean"])
        vs_std = np.array(lc["val_std"])

        ax.fill_between(ts, tm - ts_std, tm + ts_std, alpha=0.12, color=PALETTE[0])
        ax.plot(ts, tm, "o-", color=PALETTE[0], linewidth=2, markersize=4, label="Train")

        ax.fill_between(ts, vm - vs_std, vm + vs_std, alpha=0.12, color=PALETTE[1])
        ax.plot(ts, vm, "s-", color=PALETTE[1], linewidth=2, markersize=4, label="Val")

        gap = float(tm[-1] - vm[-1])
        ax.set_title(f"{model_name}\ngap={gap:+.3f}", fontsize=10)
        ax.set_xlabel("Train Size")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(lc_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Learning Curves — All Models", fontsize=14, y=1.01)
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined learning curves: {save_path}")


# ── 2. Overfitting plot (train vs val per fold) ───────────────────────────────

def plot_overfitting(
    model_name: str,
    fold_metrics: List[Dict],
    save_path: str,
):
    """
    Bar chart showing train F1 vs val F1 for each CV fold.

    A large gap on any fold indicates overfitting on that fold's training partition.
    """
    folds = [f"Fold {m['fold']}" for m in fold_metrics]
    train_f1s = [m["train_macro_f1"] for m in fold_metrics]
    val_f1s = [m["val_macro_f1"] for m in fold_metrics]

    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_train = ax.bar(x - width / 2, train_f1s, width,
                        label="Train Macro F1", color=PALETTE[0], alpha=0.85)
    bars_val = ax.bar(x + width / 2, val_f1s, width,
                      label="Val Macro F1", color=PALETTE[1], alpha=0.85)

    # Labels on bars
    for bar in bars_train:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_val:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    mean_gap = np.mean([t - v for t, v in zip(train_f1s, val_f1s)])
    ax.set_title(
        f"Train vs Val Macro F1 per Fold — {model_name}\n"
        f"Mean overfit gap: {mean_gap:+.4f}",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved overfitting plot: {save_path}")


def plot_all_overfitting(
    all_cv_results: Dict[str, dict],
    save_path: str,
):
    """
    Grid of overfitting plots for every model — one subplot per model.
    """
    n = len(all_cv_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, (model_name, results) in enumerate(all_cv_results.items()):
        ax = axes_flat[idx]
        fold_metrics = results["fold_metrics"]
        folds = [f"F{m['fold']}" for m in fold_metrics]
        train_f1s = [m["train_macro_f1"] for m in fold_metrics]
        val_f1s = [m["val_macro_f1"] for m in fold_metrics]
        x = np.arange(len(folds))
        width = 0.35

        ax.bar(x - width / 2, train_f1s, width, label="Train", color=PALETTE[0], alpha=0.85)
        ax.bar(x + width / 2, val_f1s, width, label="Val", color=PALETTE[1], alpha=0.85)

        mean_gap = np.mean([t - v for t, v in zip(train_f1s, val_f1s)])
        ax.set_title(f"{model_name}\ngap={mean_gap:+.3f}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(folds, fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=8)
        ax.set_ylim(0, 1.10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(all_cv_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Train vs Val Macro F1 per Fold — All Models", fontsize=14, y=1.01)
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined overfitting plot: {save_path}")


# ── 3. Model comparison bar chart ─────────────────────────────────────────────

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str,
):
    """
    Horizontal bar chart ranking all models by CV Macro F1.
    Error bars show ± 1 std across folds.
    """
    df = comparison_df.sort_values("mean_val_macro_f1").reset_index(drop=True)
    n = len(df)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.6 + 1)))

    bars = ax.barh(
        df["model"],
        df["mean_val_macro_f1"],
        xerr=df["std_val_macro_f1"],
        color=colors,
        alpha=0.85,
        capsize=4,
        height=0.6,
    )

    # Value labels
    for i, (val, std) in enumerate(zip(df["mean_val_macro_f1"], df["std_val_macro_f1"])):
        ax.text(val + std + 0.005, i, f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("CV Macro F1 Score")
    ax.set_title("Model Comparison — CV Macro F1 (± std)", fontsize=13)
    ax.set_xlim(0, min(1.1, df["mean_val_macro_f1"].max() + df["std_val_macro_f1"].max() + 0.12))
    ax.axvline(x=df["mean_val_macro_f1"].max(), color="gray", linestyle="--",
               linewidth=1, alpha=0.6, label="Best model")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved model comparison plot: {save_path}")


# ── 4. Confusion matrix heatmap ───────────────────────────────────────────────

def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    model_name: str,
    save_path: str,
):
    """
    Heatmap of the OOF confusion matrix with percentage annotation.
    """
    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False

    cm = cm_df.values.astype(float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100  # row-normalised %

    fig, ax = plt.subplots(figsize=(8, 6))

    if use_seaborn:
        sns.heatmap(
            cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=cm_df.columns, yticklabels=cm_df.index,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Row %"},
        )
    else:
        im = ax.imshow(cm_pct, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Row %")
        ax.set_xticks(range(len(cm_df.columns)))
        ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(cm_df.index)))
        ax.set_yticklabels(cm_df.index)
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                ax.text(j, i, f"{cm_pct[i,j]:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_pct[i, j] > 50 else "black")

    ax.set_title(f"OOF Confusion Matrix (%) — {model_name}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved confusion matrix plot: {save_path}")
