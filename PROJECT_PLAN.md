# PROJECT PLAN — MALTO Recruitment Hackathon: Text Authorship Detection

**Date:** 2026-03-16
**Competition:** Kaggle MALTO Recruitment Hackathon
**Task:** 6-class text classification (Human vs. 5 LLM families)
**Metric:** Macro F1 Score

---

## 1. Project Objective

Build a production-quality, GitHub-ready ML pipeline that classifies text samples into one of six categories:

| Label | Source          |
|-------|-----------------|
| 0     | Human-written   |
| 1     | DeepSeek        |
| 2     | Grok            |
| 3     | Claude          |
| 4     | Gemini          |
| 5     | ChatGPT         |

Maximize **Macro F1** — every class must perform well, not just the majority.

---

## 2. Problem Framing

- **Type:** Supervised multiclass text classification (6 classes)
- **Input:** Raw text strings
- **Output:** Integer label in [0, 5]
- **Key challenge:** LLM-generated text is fluent and human-like; subtle stylistic differences distinguish models
- **Hard constraint:** No external data, no pre-trained AI detectors
- **Anti-overfitting priority:** Vectorizers must be fitted inside CV folds; regularized models only

---

## 3. Dataset Assumptions

| File                  | Description                              |
|-----------------------|------------------------------------------|
| `data/train.csv`      | Columns: `text`, `label` (int 0–5)       |
| `data/test.csv`       | Column: `text` only, no labels           |
| `data/sample_submission.csv` | Columns: `ID`, `LABEL` — defines expected output format |

Assumptions:
- Text is UTF-8 encoded
- Labels are integers, not strings
- Test set has no header label column
- Class distribution may be imbalanced → Macro F1 is appropriate
- No guaranteed alignment between train row indices and test row indices

---

## 4. Repository Structure

```
malto-text-classification/
│
├── README.md                        # Full project documentation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Standard ML gitignore
├── Makefile                         # Shortcuts: make train / infer / submit
├── PROJECT_PLAN.md                  # This document
│
├── configs/
│   └── config.yaml                  # All tunable parameters, paths, flags
│
├── data/                            # Competition data (not committed to git)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── 01_eda_baseline.ipynb        # EDA only — no production logic
│
├── src/
│   ├── __init__.py
│   ├── constants.py                 # Label maps, column names, seeds
│   ├── utils.py                     # Logging, path helpers, artifact saving
│   ├── data.py                      # Safe data loading with schema validation
│   ├── preprocess.py                # Minimal configurable text cleaning
│   ├── features.py                  # TF-IDF pipeline (word + char ngrams)
│   ├── models.py                    # Model registry (LR, LinearSVC, NB)
│   ├── train.py                     # CV loop + final training logic
│   ├── evaluate.py                  # Macro F1, classification report, confusion matrix
│   ├── inference.py                 # Load model, predict on test data
│   └── submission.py                # Format and validate submission CSV
│
├── artifacts/
│   ├── models/                      # Saved joblib pipelines
│   ├── metrics/                     # cv_results.json, classification_report.txt, etc.
│   ├── submissions/                 # submission.csv outputs
│   ├── analysis/                    # Error analysis outputs
│   └── experiments/                 # Timestamped run folders
│
├── main_train.py                    # Entry point: full training pipeline
├── main_infer.py                    # Entry point: inference on test.csv
├── main_submit.py                   # Entry point: generate submission.csv
│
└── tests/
    ├── __init__.py
    ├── test_data_loading.py
    ├── test_schema_validation.py
    └── test_submission_format.py
```

---

## 5. Implementation Phases

### Phase 1 — Repository Scaffolding
- Create directory structure
- Write `requirements.txt`, `.gitignore`, `Makefile`
- Write `configs/config.yaml` with all parameters
- Write `src/constants.py` and `src/utils.py`

### Phase 2 — Data Layer
- Write `src/data.py` — load CSVs, validate schema, handle missing values
- Write `src/preprocess.py` — configurable minimal cleaning

### Phase 3 — Feature Engineering
- Write `src/features.py` — TF-IDF FeatureUnion (word ngrams + char ngrams)
- Pipeline construction that can be fitted and transformed

### Phase 4 — Model Layer
- Write `src/models.py` — model factory returning sklearn-compatible estimators
- Logistic Regression, LinearSVC, ComplementNB

### Phase 5 — Training & Validation
- Write `src/train.py` — StratifiedKFold loop, OOF collection, final fit
- Write `src/evaluate.py` — Macro F1, accuracy, classification report, confusion matrix
- Write `main_train.py` — CLI entry point wiring everything together

### Phase 6 — Inference & Submission
- Write `src/inference.py` — load pipeline, preprocess, predict
- Write `src/submission.py` — format, validate, save
- Write `main_infer.py` and `main_submit.py`

### Phase 7 — Testing
- Write `tests/test_data_loading.py`
- Write `tests/test_schema_validation.py`
- Write `tests/test_submission_format.py`

### Phase 8 — Documentation & Notebook
- Write `README.md`
- Write EDA notebook skeleton `notebooks/01_eda_baseline.ipynb`

---

## 6. File-by-File Development Plan

### `configs/config.yaml`
- Paths: data_dir, artifacts_dir
- Preprocessing flags: lowercase, normalize_unicode, strip_whitespace
- TF-IDF params: word_ngram_range, char_ngram_range, max_features, min_df, max_df
- CV params: n_splits, random_state
- Model selection: which models to run, best model name
- Training flags: run_cv, save_artifacts, verbose

### `src/constants.py`
- LABEL_MAP: {0: "Human", 1: "DeepSeek", ...}
- LABEL_NAMES: ordered list
- TRAIN_TEXT_COL, TRAIN_LABEL_COL
- TEST_TEXT_COL
- SUBMISSION_ID_COL, SUBMISSION_LABEL_COL
- RANDOM_SEED

### `src/utils.py`
- `get_logger(name)` — returns configured logger
- `load_config(path)` — loads YAML config
- `ensure_dir(path)` — mkdir -p
- `save_json(data, path)` — save metrics dict
- `get_experiment_dir(base)` — timestamped run folder
- `log_system_info()` — Python version, sklearn version

### `src/data.py`
- `load_train(path)` → DataFrame with validated schema
- `load_test(path)` → DataFrame with validated schema
- `load_sample_submission(path)` → DataFrame
- Schema checks: required columns present, no fully empty text column, label range validation

### `src/preprocess.py`
- `Preprocessor` class with `fit_transform` / `transform` methods (sklearn-compatible)
- Options: `lowercase`, `normalize_unicode`, `strip_whitespace`, `remove_repeated_spaces`
- Default: minimal cleaning only (preserve stylistic signals)

### `src/features.py`
- `build_feature_pipeline(config)` — returns sklearn Pipeline with FeatureUnion
- Word TF-IDF: ngram_range=(1,2), sublinear_tf=True
- Char TF-IDF: ngram_range=(3,5), analyzer='char_wb', sublinear_tf=True
- Combined with sparse hstack via FeatureUnion

### `src/models.py`
- `get_model(name, config)` — model factory
- LogisticRegression: C=1.0, max_iter=1000, solver='lbfgs', multi_class='auto'
- LinearSVC: C=0.1, max_iter=2000
- ComplementNB: alpha=0.1
- `get_all_models(config)` — returns dict of all models

### `src/train.py`
- `run_cross_validation(X_text, y, config)` → cv_results dict
  - Vectorizer fitted inside each fold
  - Collects OOF predictions, per-fold metrics
  - Detects high variance across folds
- `train_final_model(X_text, y, config)` → fitted pipeline
  - Full pipeline: Preprocessor → FeatureUnion(TF-IDF) → Classifier
  - Saves to artifacts/models/

### `src/evaluate.py`
- `compute_macro_f1(y_true, y_pred)` → float
- `compute_metrics(y_true, y_pred)` → dict with f1, accuracy, per-class metrics
- `generate_classification_report(y_true, y_pred)` → string
- `generate_confusion_matrix(y_true, y_pred)` → DataFrame
- `error_analysis(X_text, y_true, y_pred, proba)` → saves most-confused examples

### `src/inference.py`
- `load_pipeline(path)` → fitted sklearn pipeline
- `predict(pipeline, X_text)` → np.array of int labels [0,5]

### `src/submission.py`
- `make_submission(ids, predictions)` → DataFrame
- `validate_submission(df, sample_df)` → bool
- `save_submission(df, path)` → writes CSV

---

## 7. Preprocessing Plan

**Philosophy:** Preserve stylistic signals. LLMs have distinctive whitespace, punctuation, and capitalization patterns.

| Step                   | Default | Configurable |
|------------------------|---------|--------------|
| Unicode normalization  | Yes     | Yes          |
| Whitespace strip       | Yes     | Yes          |
| Repeated space removal | Yes     | Yes          |
| Lowercasing            | No      | Yes          |
| Punctuation removal    | No      | Yes (off)    |
| Number removal         | No      | Yes (off)    |

---

## 8. Feature Engineering Plan

**Primary features:** TF-IDF on raw (minimally cleaned) text

| Vectorizer       | Analyzer | ngram_range | max_features | Notes               |
|------------------|----------|-------------|--------------|---------------------|
| Word TF-IDF      | word     | (1, 2)      | 100,000      | sublinear_tf=True   |
| Char TF-IDF      | char_wb  | (3, 5)      | 100,000      | sublinear_tf=True   |

Combined: `FeatureUnion` → sparse concatenation → classifier

**Rationale:**
- Word ngrams capture vocabulary and phrase-level style
- Char ngrams capture punctuation patterns, spacing, morphology — very discriminative for AI vs human
- `char_wb` pads word boundaries, better for detecting LLM-style tokens
- `sublinear_tf` prevents dominant terms from overwhelming signal

---

## 9. Model Development Plan

| Model             | Key Hyperparameters     | Strength                     |
|-------------------|-------------------------|------------------------------|
| LogisticRegression| C=1.0, max_iter=1000    | Probabilistic, calibrated    |
| LinearSVC         | C=0.1, max_iter=2000    | Fast, high-dimensional text  |
| ComplementNB      | alpha=0.1               | Robust for imbalanced classes|

**Selection strategy:**
1. Run all three with StratifiedKFold
2. Compare Macro F1 across models
3. Select best model for final training and submission
4. Save comparison to `artifacts/metrics/model_comparison.csv`

---

## 10. Validation Strategy

- **Method:** Stratified K-Fold (k=5)
- **Metric:** Macro F1 (primary), Accuracy (secondary)
- **Per-fold outputs:** F1, accuracy, classification report
- **Aggregate:** Mean ± std across folds
- **OOF diagnostics:** Out-of-fold predictions for full error analysis
- **Artifacts saved:**
  - `artifacts/metrics/cv_results.json` — all fold metrics
  - `artifacts/metrics/classification_report.txt` — OOF report
  - `artifacts/metrics/confusion_matrix.csv` — OOF confusion matrix

---

## 11. Overfitting Prevention Strategy

| Safeguard                        | Implementation                         |
|----------------------------------|----------------------------------------|
| Vectorizer inside CV fold        | Pipeline rebuilt per fold              |
| Regularization                   | C parameter in LR/SVC                  |
| min_df threshold                 | Remove rare terms (noise)              |
| max_df threshold                 | Remove ubiquitous terms                |
| max_features cap                 | Prevent overfitting on vocabulary size |
| Fold variance monitoring         | Log std(F1) across folds               |
| Train vs val metric comparison   | Log both in each fold                  |
| No target leakage                | Labels never used for feature fitting  |

---

## 12. Experiment Tracking Plan

Each training run creates a timestamped directory:
```
artifacts/experiments/YYYY-MM-DD_HHMMSS/
├── config_snapshot.yaml     # Exact config used
├── cv_results.json          # CV fold metrics
├── classification_report.txt
├── confusion_matrix.csv
├── model_comparison.csv
└── run.log                  # Full training log
```

Enables reproducibility and comparison between experiments without overwriting previous runs.

---

## 13. Testing Plan

### `tests/test_data_loading.py`
- Test that train.csv loads without errors (using synthetic fixture)
- Test that test.csv loads without errors
- Test that returned DataFrames have correct shape and dtypes

### `tests/test_schema_validation.py`
- Test that missing `text` column raises ValueError
- Test that missing `label` column in train raises ValueError
- Test that out-of-range labels raise ValueError
- Test that empty text detection works

### `tests/test_submission_format.py`
- Test that submission has correct columns (ID, LABEL)
- Test that LABEL values are integers in [0, 5]
- Test that submission row count matches test set
- Test that ID column is sequential starting from 0

---

## 14. Inference and Submission Plan

**Inference flow:**
1. Load config
2. Load best trained pipeline from `artifacts/models/best_model.joblib`
3. Load `data/test.csv`
4. Apply preprocessing (same as training, via pipeline)
5. Predict labels (integer array)
6. Wrap in submission DataFrame

**Submission flow:**
1. Load predictions
2. Create DataFrame: `ID` = range(len(predictions)), `LABEL` = predictions
3. Validate against `sample_submission.csv` format
4. Save to `artifacts/submissions/submission_TIMESTAMP.csv`
5. Also save as `artifacts/submissions/submission_latest.csv`

---

## 15. GitHub Preparation Checklist

- [ ] `.gitignore` excludes `data/`, `artifacts/`, `.env`, `__pycache__`, `.ipynb_checkpoints`
- [ ] `README.md` has full setup and run instructions
- [ ] No hardcoded absolute paths anywhere
- [ ] `requirements.txt` has pinned versions
- [ ] All scripts accept `--config` CLI argument
- [ ] Sensitive data not committed
- [ ] `notebooks/` included but data not embedded in notebook
- [ ] Tests pass with synthetic data
- [ ] Artifact directories created programmatically
- [ ] Repository name: `malto-text-classification`

---

## 16. Definition of Done

The project is complete when ALL of the following are true:

- [x] `main_train.py --config configs/config.yaml` runs end-to-end without errors
- [x] Stratified K-Fold CV completes and Macro F1 is computed and logged
- [x] Model comparison table saved to `artifacts/metrics/model_comparison.csv`
- [x] Best model pipeline saved to `artifacts/models/best_model.joblib`
- [x] `main_infer.py --config configs/config.yaml` produces predictions on test set
- [x] `main_submit.py --config configs/config.yaml` produces valid `submission.csv`
- [x] Submission format validated against `sample_submission.csv`
- [x] All three test files pass
- [x] `README.md` instructions are accurate and complete
- [x] Repository has no hardcoded paths, no TODO stubs, no notebook-only logic
- [x] Experiment folder created with all artifacts for each run

---

*End of PROJECT_PLAN.md*
