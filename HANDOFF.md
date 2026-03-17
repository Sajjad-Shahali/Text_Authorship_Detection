# MALTO Hackathon - Project Handoff Document
**Date**: 2026-03-17
**Task**: Text Authorship Detection - 6-class classification, Macro F1
**Best confirmed public LB**: **0.92422**
**Best competitor**: **0.96423**
**Gap to leader**: **0.04001**
**Current local winner**: `stacking_lgbm`
**Current latest experiment dir**: `artifacts/experiments/2026-03-17_132947_run`

---

## 1. Problem Overview

Classify text as one of 6 authors:

| Label | Class    | Train samples | Test samples (est.) |
|-------|----------|---------------|---------------------|
| 0 | Human    | 1520 | ~380 |
| 1 | DeepSeek | 80   | ~20  |
| 2 | Grok     | 160  | ~40  |
| 3 | Claude   | 80   | ~20  |
| 4 | Gemini   | 240  | ~60  |
| 5 | ChatGPT  | 320  | ~80  |

**Total**: 2400 train, 600 test  
**Metric**: Macro F1  
**Core difficulty**: DeepSeek vs Grok. Both write short factual encyclopedic text and remain the bottleneck.

---

## 2. Current Status

### Best known results

| Source | Model | Score | Notes |
|--------|-------|-------|-------|
| Public leaderboard | `stacking_lgbm` submission from 2026-03-16 23:07:43 | **0.92422** | best confirmed score in repo |
| Best local CV | `stacking_lgbm` (Run 10) | **0.9393** | still the best local run |
| Latest local run | `stacking_lgbm` (Run 12) | 0.9339 | regression vs Run 10 |

### Latest completed training run

Experiment: `artifacts/experiments/2026-03-17_132947_run`

| Model | CV F1 | OOF F1 | OOF Accuracy |
|-------|------:|--------:|-------------:|
| `stacking_lgbm` | **0.9339** | **0.9342** | 0.9812 |
| `stacking_lgbm_v3` | 0.9328 | 0.9331 | 0.9808 |
| `two_stage_top2` | 0.9253 | 0.9258 | 0.9804 |

### Interpretation

- Run 12 did **not** beat Run 10.
- `stacking_lgbm_v3` was implemented successfully but did not improve over plain `stacking_lgbm`.
- The working mainline is still `stacking_lgbm`.

---

## 3. Repository Structure

```text
text-authorship-detection/
|-- configs/
|   `-- config.yaml
|-- src/
|   |-- preprocess.py
|   |-- features.py
|   |-- models.py
|   |-- train.py
|   |-- inference.py
|   |-- threshold_optimizer.py
|   |-- data.py
|   |-- evaluate.py
|   |-- submission.py
|   |-- plots.py
|   `-- utils.py
|-- data/
|   |-- train.csv
|   |-- test.csv
|   `-- sample_submission.csv
|-- artifacts/
|   |-- models/
|   |   `-- best_model.joblib
|   |-- thresholds.json
|   |-- submissions/
|   |   `-- submission_latest.csv
|   |-- metrics/
|   |   `-- model_comparison.csv
|   `-- experiments/
|       `-- YYYY-MM-DD_HHMMSS_run/
|-- EXPERIMENTS.md
|-- HANDOFF.md
`-- .env/
```

---

## 4. How to Run

### Training

```bash
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
```

### Inference

```bash
python main_infer.py --config configs/config.yaml
```

### Kaggle Submission File

- Main file: `artifacts/submissions/submission_latest.csv`
- Timestamped file example: `artifacts/submissions/submission_20260317_171925.csv`

---

## 5. Operational Notes

### Inference repair completed on 2026-03-17

`main_infer.py` was failing before model load with a YAML parser error.  
Root cause: unresolved merge/stash markers had been left in:

- `configs/config.yaml`
- `src/features.py`
- `src/models.py`

Those files were cleaned and inference was verified successfully.

### Verified inference state

- Verified command:

```bash
python main_infer.py --config configs/config.yaml
```

- Verified output file:
  - `artifacts/submissions/submission_20260317_171925.csv`
- `submission_latest.csv` is current and valid.
- Current prediction distribution on test:
  - Human: 381
  - DeepSeek: 19
  - Grok: 40
  - Claude: 20
  - Gemini: 61
  - ChatGPT: 79

### Important config behavior

`configs/config.yaml` currently sets `best_model: "stacking_lgbm_v3"`, but when CV runs, the code overrides this and trains the CV winner. In Run 12 that winner was still `stacking_lgbm`.

---

## 6. Architecture

### Pipeline

```text
Raw text
  -> Preprocessor
  -> FeatureUnion
     -> word TF-IDF
     -> char TF-IDF
     -> char micro TF-IDF
     -> stylometric features
     -> function-word TF-IDF
     -> DS/Grok subspace TF-IDF
     -> delex TF-IDF
  -> Classifier
```

### Current feature branches

- Word TF-IDF: 50k, ngram 1-2
- Char TF-IDF: 50k, ngram 2-6, `char_wb`
- Char micro TF-IDF: 20k, ngram 3-7, `char`
- Stylometric features: 49
- Function-word TF-IDF: up to 5k, ngram 1-3
- DS/Grok subspace TF-IDF: 10k, fitted only on DeepSeek/Grok samples
- Delex TF-IDF: 30k, digits replaced with `__NUM__`

Approximate total feature space: ~165k

### Key custom components

- `TwoStageClassifier`: 6-class base model plus DeepSeek/Grok specialist
- `TfidfMLPClassifier`: TruncatedSVD -> MLP
- `LGBMTfidfClassifier`: TruncatedSVD -> LightGBM
- `XGBTfidfClassifier`: TruncatedSVD -> XGBoost
- `DSGrokSubspaceTfidf`: DS/Grok-only vocabulary branch
- `DelexTfidfVectorizer`: word TF-IDF on delexicalized text
- `FunctionWordAnalyzer`: function-word n-gram extractor

---

## 7. Model Performance History

| Run | Best Model | CV F1 | LB Score | Note |
|-----|------------|------:|---------:|------|
| 1 | logistic_regression | 0.88 | - | baseline |
| 2 | logistic_regression_balanced | 0.90 | - | balanced weights |
| 3 | two_stage | 0.9280 | - | DS/Grok specialist introduced |
| 4 | two_stage | 0.9328 | 0.91089 | stylometric expansion |
| 5 | two_stage_top2 | 0.9328 | 0.91089 | top2 trigger |
| 6 | two_stage_top2 | 0.9328 | 0.91089 | more DS/Grok features |
| 7 | two_stage_top2 | 0.9328 | 0.91089 | threshold variants failed |
| 8 | ensemble_mlp | 0.9350 | 0.90978 | CV up, LB down |
| 9 | ensemble_mlp | 0.9350 | pending | calibration/DS boost work |
| 10 | **stacking_lgbm** | **0.9393** | **0.92422** | best overall run so far |
| 11 | stacking_lgbm | 0.9370 | pending | bug-fix run plus `stacking_lgbm_v2` |
| 12 | stacking_lgbm | 0.9339 | pending | new feature branches plus `stacking_lgbm_v3` |

### Best OOF report from Run 10

```text
              precision    recall  f1-score   support

       Human       1.00      1.00      1.00      1520
    DeepSeek       0.76      0.78      0.77        80
        Grok       0.89      0.90      0.90       160
      Claude       1.00      1.00      1.00        80
      Gemini       0.99      0.98      0.99       240
     ChatGPT       1.00      0.99      0.99       320
```

---

## 8. Core Problem: DeepSeek vs Grok

Root cause: both classes often produce short factual encyclopedia-style passages.

Current best-model confusion from Run 10:

- DS -> Grok: 17
- Grok -> DS: 15
- Total pair confusion: 32

What has already been tried:

- pair-specific DS/Grok thresholding
- extra DeepSeek weight
- two-stage trigger variants
- MLP non-linear boundaries
- LightGBM + stacking
- DS/Grok-only vocabulary branch
- delexicalized TF-IDF
- XGBoost as a fourth stack member

What the latest evidence says:

- `stacking_lgbm` is still the strongest path
- `stacking_lgbm_v3` did not improve over the base stack
- broad model branching is currently less promising than tighter calibration or smarter data use

---

## 9. Recommended Next Steps

### Keep following

1. Submit the repaired current `stacking_lgbm` inference output if it has not been submitted yet.
2. Continue from the `stacking_lgbm` line, not from `two_stage_top2` or `stacking_lgbm_v3`.
3. Focus on calibration and data leverage rather than bigger architecture branching.

### Highest-value experiments next

1. **Pseudo-labeling**
   - Use high-confidence test predictions only.
   - Prefer consensus between two strong `stacking_lgbm` variants.
   - Keep pseudo-label sample weights low, for example 0.3.

2. **Threshold tuning using expected DS:Grok ratio**
   - Current DS/Grok pair threshold is `0.65`.
   - Revisit this with the observed test prediction distribution and class prior assumptions.

3. **Feature regularization / pruning**
   - Run 12 added useful ideas but still regressed overall.
   - The next iteration should isolate which new branch hurts calibration rather than adding more branches at once.

### Lower priority now

- more two-stage trigger variants
- more MLP variants
- promoting `stacking_lgbm_v3` as the new default

### Competition constraint

Pre-trained AI-generated-text detectors are forbidden.  
Do not use transformer fine-tuning, sentence-transformer embeddings, or external pretrained detectors.

---

## 10. Key Config Settings

```yaml
models:
  run_models: [two_stage_top2, stacking_lgbm, stacking_lgbm_v3]
  best_model: "stacking_lgbm_v3"   # overridden by CV winner when run_cv=true
  two_stage_margin_gap: 0.40
  two_stage_ds_threshold: 0.52

  lgbm:
    n_svd_components: 500
    n_estimators: 500
    num_leaves: 63
    learning_rate: 0.05

  xgb:
    n_svd_components: 300
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05

features:
  function_word_tfidf:
    enabled: true
    ngram_range: [1, 3]
    max_features: 5000

  ds_grok_tfidf:
    enabled: true
    ngram_range: [1, 3]
    max_features: 10000

  delex_tfidf:
    enabled: true
    ngram_range: [1, 2]
    max_features: 30000
```

---

## 11. Known Bugs Fixed

| Bug | Status |
|-----|--------|
| `StyleometricTransformer._f()` empty-text vector size mismatch | fixed |
| `import math` inside per-sample feature extraction | fixed |
| `margin_trigger_gap` read but not passed in earlier stack paths | fixed |
| Preprocessor collapsing `\n` and destroying markdown structure | fixed |
| `ensemble_mlp` hardcoded `binary_ds_threshold=0.50` | fixed |
| unresolved merge markers blocking inference in config/model files | fixed on 2026-03-17 |

---

## 12. Current Thresholds

File: `artifacts/thresholds.json`

```json
{
  "model": "stacking_lgbm",
  "thresholds": [1.0, 0.5, 1.25, 1.0, 1.25, 1.0],
  "ds_grok_pair_threshold": 0.65
}
```

Notes:

- `thresholds[i]` scales class probability before argmax
- `0.5` on DeepSeek biases downward against false DeepSeek predictions
- `1.25` on Grok and Gemini slightly biases upward
- `0.65` makes DeepSeek prediction more conservative in the DS/Grok pair

---

## 13. Environment

```text
Python 3.12.10
scikit-learn 1.8.0
lightgbm 4.6.0
xgboost 3.2.0
numpy 2.4.3
pandas 3.0.1
scipy 1.17.1
joblib 1.5.3
Platform: Windows 11 Pro
```

Virtual environment: `.env/`  
Activate with: `.env\Scripts\activate`
