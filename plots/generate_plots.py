"""Generate LinkedIn plots for MALTO Hackathon — all data from real artifacts."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs(os.path.dirname(__file__), exist_ok=True)
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

BLUE   = '#2563EB'
GREEN  = '#16A34A'
RED    = '#DC2626'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
TEAL   = '#0891B2'
GRAY   = '#6B7280'
DARK   = '#111827'


# ─── 1. CONFUSION MATRIX ────────────────────────────────────────────────────
labels = ['Human', 'DeepSeek', 'Grok', 'Claude', 'Gemini', 'ChatGPT']
cm = np.array([
    [1519,   1,   0,  0,   0,   0],
    [   0,  62,  17,  0,   1,   0],
    [   0,  15, 144,  0,   0,   1],
    [   0,   0,   0, 80,   0,   0],
    [   1,   4,   0,  0, 235,   0],
    [   3,   0,   0,  0,   1, 316],
])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted', fontsize=12, labelpad=10)
ax.set_ylabel('Actual',    fontsize=12, labelpad=10)
ax.set_title(
    'Confusion Matrix — stacking_lgbm  (OOF, 5-fold CV)\nMacro F1 = 0.9393',
    fontsize=13, fontweight='bold', pad=15
)
for i in range(6):
    for j in range(6):
        v = cm[i, j]; p = cm_norm[i, j]
        if v > 0:
            color = 'white' if p > 0.6 else DARK
            ax.text(j, i, f'{v}\n({p:.0%})', ha='center', va='center',
                    fontsize=9, color=color,
                    fontweight='bold' if i == j else 'normal')
plt.colorbar(im, ax=ax, label='Recall Rate', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '01_confusion_matrix.png'), dpi=150)
plt.close()
print('saved: 01_confusion_matrix.png')


# ─── 2. PER-CLASS F1 ────────────────────────────────────────────────────────
classes  = ['Human', 'DeepSeek', 'Grok', 'Claude', 'Gemini', 'ChatGPT']
f1       = [1.00,    0.77,       0.90,   1.00,     0.99,     0.99]
support  = [1520,    80,         160,    80,        240,      320]
colors   = [GREEN if s >= 0.95 else (ORANGE if s >= 0.85 else RED) for s in f1]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(classes, f1, color=colors, edgecolor='white', linewidth=1.5, zorder=2)
ax.set_ylim(0.60, 1.10)
avg = np.mean(f1)
ax.axhline(avg, color=BLUE, linestyle='--', linewidth=1.8,
           label=f'Macro avg = {avg:.2f}', zorder=3)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title(
    'Per-class F1 Score — stacking_lgbm (OOF)\nDeepSeek & Grok are the bottleneck',
    fontsize=13, fontweight='bold', pad=12
)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

for bar, score, n in zip(bars, f1, support):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.text(bar.get_x() + bar.get_width()/2, 0.63,
            f'n={n}', ha='center', va='bottom', fontsize=8.5, color=GRAY)

legend_handles = [
    mpatches.Patch(color=GREEN,  label='F1 ≥ 0.95'),
    mpatches.Patch(color=ORANGE, label='F1 0.85–0.95'),
    mpatches.Patch(color=RED,    label='F1 < 0.85'),
    plt.Line2D([0], [0], color=BLUE, linestyle='--',
               label=f'Macro avg = {avg:.2f}'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '02_per_class_f1.png'), dpi=150)
plt.close()
print('✓ 02_per_class_f1.png')


# ─── 3. MODEL COMPARISON ────────────────────────────────────────────────────
model_labels = ['LGBM+SVD', 'MLP+SVD\n(calibrated)', 'TwoStage\n(top2)',
                'Ensemble\nMLP', 'Ensemble\nLGBM', 'Stacking\nLGBM ★']
cv_f1  = [0.9239, 0.9256, 0.9303, 0.9305, 0.9334, 0.9393]
oof_f1 = [0.9237, 0.9263, 0.9313, 0.9314, 0.9338, 0.9397]

x = np.arange(len(model_labels))
w = 0.35
bar_colors = [BLUE] * 5 + [GREEN]

fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - w/2, cv_f1,  w, color=bar_colors, alpha=0.90, edgecolor='white', label='CV F1 (mean 5-fold)')
b2 = ax.bar(x + w/2, oof_f1, w, color=bar_colors, alpha=0.45, edgecolor='white', label='OOF F1')

ax.set_ylabel('Macro F1', fontsize=12)
ax.set_title(
    'Model Comparison — CV F1 vs OOF F1\nStacking outperforms all single-model approaches',
    fontsize=13, fontweight='bold', pad=12
)
ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylim(0.910, 0.948)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.legend(fontsize=10)

for bar, val in zip(b1, cv_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, '03_model_comparison.png'), dpi=150)
plt.close()
print('✓ 03_model_comparison.png')


# ─── 4. CLASS DISTRIBUTION ──────────────────────────────────────────────────
class_names = ['Human\n(0)', 'DeepSeek\n(1)', 'Grok\n(2)',
               'Claude\n(3)', 'Gemini\n(4)', 'ChatGPT\n(5)']
counts = [1520, 80, 160, 80, 240, 320]
total  = sum(counts)
pcts   = [c / total * 100 for c in counts]
clrs   = ['#1D4ED8', RED, ORANGE, PURPLE, GREEN, TEAL]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(class_names, counts, color=clrs, edgecolor='white', linewidth=1.5, zorder=2)
ax.set_ylabel('Training Samples', fontsize=12)
ax.set_title(
    'Dataset Class Distribution — Severe Imbalance\n2,400 total training samples · 6 classes',
    fontsize=13, fontweight='bold', pad=12
)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)

for bar, n, p in zip(bars, counts, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{n}\n({p:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('Only 80 samples!\nHardest to classify',
            xy=(1, 80), xytext=(2.3, 600),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUT, '04_class_distribution.png'), dpi=150)
plt.close()
print('✓ 04_class_distribution.png')


# ─── 5. LEADERBOARD PROGRESSION ─────────────────────────────────────────────
runs   = [1,   4,       5,       6,            7,       8,            10,        16,     17,      18]
lb     = [0.85942, 0.91089, 0.91089, 0.91089, 0.91089, 0.90978, 0.92422, 0.91760, 0.90430, 0.90984]
rlabels = [
    'Run 1\nBaseline',
    'Run 4\n+DS/Grok\nSpecialist',
    'Run 5',
    'Run 6\n+43 Style\nFeatures',
    'Run 7',
    'Run 8\n+MLP\n(LB↓)',
    'Run 10\nStacking ★',
    'Run 16',
    'Run 17',
    'Run 18',
]

xi = range(len(runs))
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(xi, lb, 'o-', color=BLUE, linewidth=2.2, markersize=7, zorder=3)
ax.fill_between(xi, lb, min(lb) - 0.005, alpha=0.07, color=BLUE)

best_i = lb.index(max(lb))
ax.scatter([best_i], [lb[best_i]], color=GREEN, s=220, zorder=5,
           label=f'Best LB: {max(lb)} (Run {runs[best_i]})')

ax.axhline(0.96423, color=RED, linestyle='--', linewidth=1.5,
           label='Best competitor: 0.96423')

ax.set_xticks(xi); ax.set_xticklabels(rlabels, fontsize=8.5)
ax.set_ylabel('Public Leaderboard Macro F1', fontsize=12)
ax.set_title(
    'Leaderboard Score Progression — 18 Experiment Runs\n0.859 → 0.924  (+6.4 pp over baseline)',
    fontsize=13, fontweight='bold', pad=12
)
ax.set_ylim(0.835, 0.978)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.legend(fontsize=10, loc='lower right')

ax.annotate(f'0.92422\n(Run 10)',
            xy=(best_i, lb[best_i]),
            xytext=(best_i - 1.6, lb[best_i] + 0.013),
            fontsize=9, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUT, '05_lb_progression.png'), dpi=150)
plt.close()
print('✓ 05_lb_progression.png')


# ─── 6. FEATURES + CV STABILITY ─────────────────────────────────────────────
feat_names = ['Word TF-IDF\n(bigrams)', 'Char TF-IDF\n(2–6 grams)',
              'Stylometric\n(43 features)', 'Function-word\nTF-IDF (151)']
feat_dims  = [50000, 50000, 43, 151]
feat_clrs  = ['#3B82F6', '#8B5CF6', '#F59E0B', '#10B981']

fold_names = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
fold_vals  = [0.9176,   0.9581,   0.9448,   0.9500,   0.9261]
fold_mean  = np.mean(fold_vals)
fold_std   = np.std(fold_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left — feature dims (log)
bars_l = ax1.barh(feat_names, feat_dims, color=feat_clrs, edgecolor='white',
                  linewidth=1.5, height=0.5)
ax1.set_xscale('log')
ax1.set_xlabel('Feature Dimensions (log scale)', fontsize=11)
ax1.set_title('Feature Space Breakdown\n~100,000 total dimensions', fontsize=12, fontweight='bold')
ax1.set_xlim(10, 300000)
for bar, dim in zip(bars_l, feat_dims):
    ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
             f'{dim:,}', va='center', fontsize=10, fontweight='bold')

# Right — CV fold stability
fold_clrs = [GREEN if v >= fold_mean else BLUE for v in fold_vals]
bars_r = ax2.bar(fold_names, fold_vals, color=fold_clrs, edgecolor='white',
                 linewidth=1.5, zorder=2)
ax2.axhline(fold_mean, color=RED, linestyle='--', linewidth=1.8,
            label=f'Mean = {fold_mean:.4f}')
ax2.set_ylim(0.890, 0.970)
ax2.set_ylabel('Validation Macro F1', fontsize=11)
ax2.set_title(
    f'stacking_lgbm — 5-Fold CV Stability\nMean={fold_mean:.4f} ± {fold_std:.4f}',
    fontsize=12, fontweight='bold'
)
ax2.yaxis.grid(True, alpha=0.3); ax2.set_axisbelow(True)
ax2.legend(fontsize=9)
for bar, val in zip(bars_r, fold_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Feature Engineering & Cross-Validation Stability', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_features_and_cv.png'), dpi=150)
plt.close()
print('✓ 06_features_and_cv.png')


print('\nAll 6 plots saved to plots/')
