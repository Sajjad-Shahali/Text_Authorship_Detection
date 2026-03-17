# MALTO Hackathon — Project Handoff Document
**Date**: 2026-03-17
**Task**: Text Authorship Detection — 6-class classification, Macro F1
**Competition status**: Best LB = **0.91089** | Best competitor = **0.96423** | Gap = 0.052
**Current branch**: Run 12 implemented, not yet submitted

---

## 1. Problem Overview

Classify text as one of 6 authors:

| Label | Class    | Train samples | Test samples (est.) |
|-------|----------|---------------|---------------------|
| 0     | Human    | 1520          | ~380                |
| 1     | DeepSeek | 80            | ~20                 |
| 2     | Grok     | 160           | ~40                 |
| 3     | Claude   | 80            | ~20                 |
| 4     | Gemini   | 240           | ~60                 |
| 5     | ChatGPT  | 320           | ~80                 |

**Total**: 2400 train, 600 test
**Metric**: Macro F1 (all classes weighted equally)
**Core difficulty**: DeepSeek (80 samples) vs Grok (160 samples) — both produce short factual encyclopedic sentences, visually indistinguishable

---

## 2. Repository Structure

```
text-authorship-detection/
├── configs/
│   └── config.yaml              # All hyperparameters and model selection
├── src/
│   ├── preprocess.py            # Text cleaning (preserves \n as of Run 11)
│   ├── features.py              # FeatureUnion: word/char/micro-char/stylometric/function-word TF-IDF
│   ├── models.py                # All model classes + factory function get_model()
│   ├── train.py                 # CV logic, model comparison, threshold optimization
│   ├── inference.py             # Predict on test set using trained model
│   ├── threshold_optimizer.py   # Per-class + DS/Grok pair threshold optimization
│   ├── data.py                  # Data loading
│   ├── evaluate.py              # F1 metrics
│   ├── submission.py            # Submission file generation
│   ├── plots.py                 # Learning curve / confusion matrix plots
│   └── utils.py                 # Logger
├── main_train.py                # Training entry point
├── main_infer.py                # Inference entry point
├── data/
│   ├── train.csv                # TEXT, LABEL columns
│   ├── test.csv                 # TEXT column only
│   └── sample_submission.csv
├── artifacts/
│   ├── models/
│   │   └── best_model.joblib    # Latest trained model
│   ├── thresholds.json          # Per-class + DS/Grok pair thresholds
│   ├── submissions/
│   │   └── submission_latest.csv
│   ├── metrics/
│   │   └── model_comparison.csv # All-time model comparison
│   └── experiments/
│       └── YYYY-MM-DD_HHMMSS_run/
│           ├── model_comparison.csv
│           ├── thresholds.json
│           ├── config_snapshot.yaml
│           ├── best_model.joblib
│           ├── run.log
│           └── <model_name>/
│               ├── best_fold_model.joblib
│               ├── classification_report.txt
│               ├── confusion_matrix.csv
│               ├── error_analysis.csv
│               └── submission_*.csv  ← per-model submission
├── EXPERIMENTS.md               # Full run log with results and diagnosis
├── HANDOFF.md                   # This document
└── .env/                        # Python virtual environment
```

---

## 3. How to Run

### Training (produces model + submission)
```bash
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
```

### Inference (re-run on test set with best model + thresholds)
```bash
python main_infer.py --config configs/config.yaml
```

### Submit to Kaggle
- **Best model submission**: `artifacts/submissions/submission_latest.csv`
- **Per-model comparison**: each `artifacts/experiments/<run>/<model>/submission_*.csv`

---

## 4. Architecture

### Pipeline (anti-leakage: fully rebuilt inside each CV fold)
```
Raw text
  └─ Preprocessor (normalize unicode, strip whitespace, preserve \n)
       └─ FeatureUnion
            ├─ word TF-IDF           (50k features, ngram 1-2, word analyzer)
            ├─ char TF-IDF           (50k features, ngram 2-6, char_wb analyzer)
            ├─ char micro TF-IDF     (20k features, ngram 3-7, char analyzer)    ← Run 11
            ├─ StyleometricPipeline  (49 hand-crafted features + MaxAbsScaler)   ← Run 12
            ├─ function-word TF-IDF  (5k features, fw bigrams/trigrams)          ← Run 12
            ├─ DS/Grok subspace TF-IDF (10k features, fitted on DS+Grok only)   ← Run 12
            └─ delex TF-IDF          (30k features, digits → __NUM__)            ← Run 12
               └─ Classifier
```
**Total feature space: ~165k**

### 49 Stylometric Features (key ones)
- Text length, word count, sentence count, paragraph count
- Avg word length, type-token ratio, long-word ratio
- Very short/long sentence ratios, avg para chars
- Punctuation rates: comma, period, exclamation, question, colon, semicolon, dash, ellipsis
- Markdown rates: bullet, numbered list, header, code, bold, italic, link
- `starts_with_the`, `clause_per_sent` (Run 10)
- `first_sent_words`, `proper_noun_density`, `hedge_rate`, `question_per_sent`, `sent_range`, `text_len_log` (Run 6)
- `punct_variety`, `sent_length_cv`, `transition_word_rate` (Run 3)
- `def_opening`, `article_rate`, `copula_rate`, `year_rate` (Run 12 NEW)

### Key Model Classes

**`TwoStageClassifier`** (`src/models.py:200`)
6-class LR base → binary DS/Grok specialist. Trigger: `top2_trigger` (DS+Grok both top-2) AND `margin_trigger_gap` (|P(DS)-P(Grok)| < 0.40). Threshold `binary_ds_threshold` controls DS conservatism.

**`TfidfMLPClassifier`** (`src/models.py:~130`)
`TruncatedSVD(500 dims)` → `MLPClassifier(512, 256, ReLU)`. Breaks the linear TF-IDF ceiling.

**`LGBMTfidfClassifier`** (`src/models.py`)
`TruncatedSVD(500 dims)` → `LGBMClassifier`. Native class-weight support, tree-based non-linear interactions.

**`XGBTfidfClassifier`** (`src/models.py`) ← Run 12 NEW
`TruncatedSVD(300 dims)` → `XGBClassifier`. Depth-wise tree growth (different from LGBM's leaf-wise). Adds ensemble diversity.

**`DSGrokSubspaceTfidf`** (`src/features.py`) ← Run 12 NEW
TF-IDF fitted only on DS+Grok training samples per fold. Topic-neutral DS/Grok vocabulary.

**`DelexTfidfVectorizer`** (`src/features.py`) ← Run 12 NEW
Word TF-IDF on text with `\d+` → `__NUM__`. Reduces year/number topic leakage.

**`FunctionWordAnalyzer`** (`src/features.py`) ← Run 12 NEW
Callable analyzer extracting n-grams of function words only. Replaces static unigram vocabulary.

**`StackingClassifier`** (sklearn)
- `stacking_lgbm`: 18-dim meta-features (3 bases × 6 classes) → LR meta
- `stacking_lgbm_v3`: 24-dim meta-features (4 bases × 6 classes) → LR meta, `StratifiedKFold(5)` inner CV ← Run 12 NEW

---

## 5. Model Performance History

| Run | Best Model         | CV F1  | LB Score | Key Change |
|-----|--------------------|--------|----------|------------|
| 1   | logistic_regression | 0.88   | —        | Baseline   |
| 2   | logistic_regression_balanced | 0.90 | — | Balanced weights |
| 3   | two_stage          | 0.9280 | —        | DS/Grok binary specialist |
| 4   | two_stage          | 0.9328 | 0.91089  | 43 stylometric features |
| 5   | two_stage_top2     | 0.9328 | 0.91089  | top2 trigger, pair threshold |
| 6   | two_stage_top2     | 0.9328 | 0.91089  | 6 new DS/Grok features, bug fix |
| 7   | two_stage_top2     | 0.9328 | 0.91089  | threshold variants (all failed) |
| 8   | ensemble_mlp       | 0.9350 | 0.90978  | MLP broke linear ceiling (LB dropped) |
| 9   | ensemble_mlp       | 0.9350 | pending  | MLP config fix, DS boost, calibration |
| 10  | **stacking_lgbm**  | **0.9393** | pending | LightGBM + stacking meta-learner |
| 11  | stacking_lgbm (fixed) | pending | pending | Bug fixes (newline, margin_trigger_gap) + stacking_lgbm_v2 |
| 12  | stacking_lgbm_v3   | pending | pending  | 4 new features, 3 new TF-IDF branches, XGBoost 4th base |

### Current Best (Run 10 — OOF classification report)
```
              precision    recall  f1-score   support

       Human       1.00      1.00      1.00      1520
    DeepSeek       0.76      0.78      0.77        80   ← BOTTLENECK
        Grok       0.89      0.90      0.90       160
      Claude       1.00      1.00      1.00        80
      Gemini       0.99      0.98      0.99       240
     ChatGPT       1.00      0.99      0.99       320

    accuracy                           0.98      2400
   macro avg       0.94      0.94      0.94      2400
```

---

## 6. The Core Problem: DeepSeek vs Grok

**Root cause**: Both produce short, 1-2 sentence factual encyclopedic texts.
Example DS→Grok error: *"Virtual reality immerses users in completely digital environments..."*
Example Grok→DS error: *"The ancient Silk Road was a network of trade routes..."*

Both sets are ~191 chars / ~27 words — statistically identical. The classifier cannot separate them based on length, vocabulary, or structure for these short factual texts.

**What has been tried:**
- Pair-specific DS/Grok threshold (ratio P(DS)/(P(DS)+P(Grok))) — small improvement
- DeepSeek sample weight boost — DS recall up but Grok→DS exploded
- TwoStageClassifier with various trigger modes — small gains, trade-off ceiling
- MLP non-linear boundaries — CV improved, LB dropped (calibration issue)
- LightGBM + stacking — **currently best** at 0.9393 CV

**Current DS/Grok confusion (best model):**
DS→Grok: 17 errors | Grok→DS: 15 errors | Total: 32 (down from 35 baseline)

---

## 7. Run 12 — What Was Just Implemented (Not Yet Run)

All code is in place. Just run `python main_train.py`:

| Change | File | Expected impact |
|--------|------|-----------------|
| Bug: `[0.0]*43` → `[0.0]*49` | `src/features.py:103` | Fixes silent shape mismatch on empty texts |
| `import math` moved to module level | `src/features.py:1` | Minor perf (no functional impact) |
| 4 new stylometric features | `src/features.py` | +0.003–0.008 CV on DS/Grok |
| Function-word bigrams/trigrams | `src/features.py` | Richer style fingerprint |
| `DSGrokSubspaceTfidf` branch | `src/features.py` | Topic-neutral DS/Grok vocabulary |
| `DelexTfidfVectorizer` branch | `src/features.py` | Reduces year/number topic leakage |
| `stacking_lgbm_v3` | `src/models.py` | XGBoost diversity + StratifiedKFold |

**Run 12 models**: `two_stage_top2` (reference), `stacking_lgbm` (baseline), `stacking_lgbm_v3` (new best candidate)

---

## 8. Future Improvement Directions

### Already done (Run 12)
- ✅ DS/Grok focused sub-vocabulary TF-IDF (`DSGrokSubspaceTfidf`)
- ✅ XGBoost as 4th stacking base (`stacking_lgbm_v3`)
- ✅ Function-word bigrams/trigrams (`FunctionWordAnalyzer`)
- ✅ Delexicalized TF-IDF (`DelexTfidfVectorizer`)

### Next (sklearn, no GPU, no pre-trained models)
1. **Pseudo-labeling**: Use high-confidence test predictions (`max_proba > 0.99`) as
   pseudo-labels. Use both `stacking_lgbm` and `stacking_lgbm_v3` agreeing (consensus).
   Retrain with pseudo-label sample weight 0.3. Especially useful for Human (dominant class).
   Use `P(DS)/(P(DS)+P(Grok)) > 0.90` gate for DS pseudo-labels (strict to avoid reinforcing noise).

2. **Threshold grid search using test class ratios**: The test set has ~1:2 DS:Grok ratio
   (estimated from sample_submission). Use this as a prior to calibrate `ds_grok_pair_threshold`
   more accurately than pure OOF optimization.

3. **Character n-gram range expansion**: Try `(2,8)` or `(3,8)` for char_tfidf_micro —
   longer char n-grams may capture sentence-level punctuation cadence patterns.

4. **`colsample_bytree` tuning for LGBM**: Currently 0.8. Try 0.6 to force the trees to
   use the new DS/Grok sub-vocabulary features more — otherwise word_tfidf may dominate.

### Constraint reminder
**Pre-trained AI-generated text detectors are forbidden** (competition rule).
This rules out: transformer fine-tuning, sentence-transformers, GPTZero-style detectors.
All features must be trained from scratch on the competition training data.

---

## 9. Key Config Settings

```yaml
# configs/config.yaml — current Run 12 settings

models:
  run_models: [two_stage_top2, stacking_lgbm, stacking_lgbm_v3]
  best_model: "stacking_lgbm_v3"       # overridden by CV winner
  two_stage_margin_gap: 0.40
  two_stage_ds_threshold: 0.52

  lgbm:
    n_svd_components: 500
    n_estimators: 500
    num_leaves: 63
    learning_rate: 0.05

  xgb:                                 # NEW Run 12
    n_svd_components: 300
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    reg_alpha: 0.1
    reg_lambda: 1.0

features:
  function_word_tfidf:
    enabled: true
    ngram_range: [1, 3]                # NEW Run 12: was unigrams only
    max_features: 5000

  ds_grok_tfidf:                       # NEW Run 12
    enabled: true
    ngram_range: [1, 3]
    max_features: 10000

  delex_tfidf:                         # NEW Run 12
    enabled: true
    ngram_range: [1, 2]
    max_features: 30000
```

---

## 10. Important Known Bugs Fixed

| Bug | Run found | Fix applied |
|-----|-----------|-------------|
| `StyleometricTransformer._f()` returned `[0.0]*43` for empty texts, actual size was 45 (→49) | Run 12 (Codex) | Run 12 |
| `import math` inside per-sample `_f()` — redundant overhead | Run 12 (Codex) | Run 12 |
| `margin_trigger_gap` read from config but never passed to `TwoStageClassifier` | Run 10 (Codex) | Run 11 |
| Preprocessor collapsed `\n` → space, destroying Gemini markdown structure | Run 10 (Codex) | Run 11 |
| `ensemble_mlp` hardcoded `binary_ds_threshold=0.50` instead of reading config | Run 8 (GPT) | Run 9 |
| `TwoStageClassifier` didn't fetch `base_proba` for `top2_trigger` path | Run 6 | Run 6 |
| Duplicate function words (`whereas`, `neither`, `nor`) in vocabulary | Run 8 | Run 8 |
| sklearn 1.8 `is_classifier()` failure: wrong MRO in `TwoStageClassifier` | Run 8 | Run 8 |

---

## 11. Thresholds

Saved to `artifacts/thresholds.json` after each run:

```json
{
  "model": "stacking_lgbm",
  "thresholds": [2.25, 0.75, 1.0, 1.0, 2.5, 1.75],
  "ds_grok_pair_threshold": 0.55
}
```

- `thresholds[i]`: scale factor on P(class_i) before argmax (>1 = bias toward this class)
- `thresholds[1]=0.75`: bias DOWN for DeepSeek (reduce false DS predictions from aggressive LGBM)
- `thresholds[4]=2.5`: bias UP for Gemini (counteract Gemini→DS bleed from Run 10)
- `ds_grok_pair_threshold=0.55`: P(DS)/(P(DS)+P(Grok)) must exceed 0.55 to predict DS when both classes are active — more conservative DS prediction

---

## 12. Environment

```
Python 3.12.10
scikit-learn 1.8.0
lightgbm     4.6.0
xgboost      3.2.0
numpy        2.4.3
pandas       3.0.1
scipy        1.17.1
joblib       1.5.3
Platform: Windows 11 Pro
```

Virtual environment: `.env/` (project root)
Activate: `.env\Scripts\activate`
