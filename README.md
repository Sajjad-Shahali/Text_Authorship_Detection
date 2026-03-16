# MALTO Recruitment Hackathon — Text Authorship Detection

> **Competition:** Kaggle MALTO Recruitment Hackathon
> **Task:** 6-class multiclass text classification
> **Metric:** Macro F1 Score
> **Constraint:** Only provided dataset; no external data; no pre-trained AI detectors

---

## Competition Overview

The goal is to classify each text sample into one of six categories:

| Label | Source       |
|-------|--------------|
| 0     | Human-written |
| 1     | DeepSeek      |
| 2     | Grok          |
| 3     | Claude        |
| 4     | Gemini        |
| 5     | ChatGPT       |

Evaluation is by **Macro F1 Score** — every class must perform well. Accuracy alone is insufficient.

---

## Project Structure

```
malto-text-classification/
│
├── configs/config.yaml          # All parameters (paths, models, features)
├── data/                        # Competition data (not in git)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── constants.py             # Label maps, column names
│   ├── utils.py                 # Logging, config loading, artifact helpers
│   ├── data.py                  # Safe data loading with schema validation
│   ├── preprocess.py            # Minimal configurable text cleaning
│   ├── features.py              # TF-IDF feature pipeline (word + char ngrams)
│   ├── models.py                # Model factory (LR, LinearSVC, ComplementNB)
│   ├── train.py                 # CV loop + final training
│   ├── evaluate.py              # Macro F1, classification report, confusion matrix
│   ├── inference.py             # Load pipeline, predict on test
│   └── submission.py            # Format and validate submission CSV
│
├── artifacts/
│   ├── models/                  # Saved joblib pipelines
│   ├── metrics/                 # CV results, classification reports
│   ├── submissions/             # Kaggle submission CSVs
│   ├── analysis/                # Error analysis outputs
│   └── experiments/             # Timestamped run folders
│
├── main_train.py                # Training entry point
├── main_infer.py                # Inference entry point
├── main_submit.py               # Submission generator entry point
│
├── tests/                       # Unit tests (synthetic data)
│   ├── test_data_loading.py
│   ├── test_schema_validation.py
│   └── test_submission_format.py
│
├── notebooks/
│   └── 01_eda_baseline.ipynb    # EDA (no production logic)
│
├── requirements.txt
├── Makefile
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/malto-text-classification.git
cd malto-text-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add competition data

Download from Kaggle and place in the `data/` directory:

```
data/train.csv
data/test.csv
data/sample_submission.csv
```

---

## Training Instructions

Run the full training pipeline (cross-validation + final model):

```bash
python main_train.py --config configs/config.yaml
```

Or use the Makefile shortcut:

```bash
make train
```

**What happens:**
1. Config is loaded
2. A timestamped experiment directory is created under `artifacts/experiments/`
3. Training data is loaded and validated
4. All configured models run 5-fold stratified cross-validation
5. Model comparison table is saved to `artifacts/metrics/model_comparison.csv`
6. Final pipeline is trained on all data and saved to `artifacts/models/best_model.joblib`

---

## Inference Instructions

```bash
python main_infer.py --config configs/config.yaml
```

Or:

```bash
make infer
```

Loads `artifacts/models/best_model.joblib`, runs it on `data/test.csv`, and saves predictions.

---

## Submission Instructions

```bash
python main_submit.py --config configs/config.yaml
```

Or:

```bash
make submit
```

Generates and validates `artifacts/submissions/submission_latest.csv`.
Upload this file to the Kaggle competition page under "Submit Predictions".

---

## Running Tests

```bash
make test
```

Or with coverage:

```bash
make test-cov
```

Tests use synthetic data and do not require competition files to run.

---

## Evaluation Metric

**Macro F1 Score** = arithmetic mean of per-class F1 scores.

Each class contributes equally regardless of frequency. A model that ignores a minority class will score poorly even if overall accuracy is high.

```
Macro F1 = mean(F1_class_0, F1_class_1, ..., F1_class_5)
```

---

## Feature Engineering

The pipeline combines two TF-IDF representations:

| Vectorizer   | Analyzer | ngram_range | Key signal                              |
|--------------|----------|-------------|-----------------------------------------|
| Word TF-IDF  | word     | (1, 2)      | Vocabulary, phrase-level style          |
| Char TF-IDF  | char_wb  | (3, 5)      | Punctuation, spacing, morphology        |

Character-level features are particularly discriminative for detecting LLM vs. human text and distinguishing between AI families.

---

## Overfitting Prevention Strategy

| Safeguard                    | Implementation                            |
|------------------------------|-------------------------------------------|
| Vectorizer inside CV folds   | Full pipeline rebuilt per fold            |
| Regularization               | `C` parameter in LR and LinearSVC         |
| Vocabulary size cap          | `max_features=100000` in TF-IDF           |
| Term frequency thresholds    | `min_df=2`, `max_df=0.95`                 |
| Stratified K-Fold            | Class balance preserved across folds      |
| Train vs. val F1 monitoring  | Logged per fold, gap flagged if > 0.10    |

---

## Kaggle Environment

To run on Kaggle, set the environment mode in `config.yaml`:

```yaml
environment:
  mode: "kaggle"
  kaggle_input_dir: "/kaggle/input/malto-hackathon"
  kaggle_working_dir: "/kaggle/working"
```

Paths are automatically overridden when `mode: kaggle` is set.

---

## Limitations

- TF-IDF features cannot capture semantic meaning or long-range coherence
- Character ngrams may vary in effectiveness across different text lengths
- ComplementNB assumes feature independence (violated in practice)
- Final model is trained on full training data — no held-out test for model selection

---

## Future Improvements

- Fine-tune a transformer (DeBERTa, RoBERTa) as a stronger backbone
- Ensemble TF-IDF classifiers with neural model predictions
- Apply class-weighted loss to address potential class imbalance
- Calibrate LinearSVC probabilities using Platt scaling (CalibratedClassifierCV)
- Pseudo-labeling on high-confidence test predictions (if competition rules allow)
- Hyperparameter search with Optuna or GridSearchCV

---

## Artifacts Reference

| Path                                          | Contents                         |
|-----------------------------------------------|----------------------------------|
| `artifacts/models/best_model.joblib`          | Final trained pipeline           |
| `artifacts/metrics/model_comparison.csv`      | CV results for all models        |
| `artifacts/metrics/cv_results.json`           | Per-fold metrics                 |
| `artifacts/metrics/classification_report.txt`| OOF classification report        |
| `artifacts/metrics/confusion_matrix.csv`      | OOF confusion matrix             |
| `artifacts/analysis/error_analysis.csv`       | Top misclassified examples       |
| `artifacts/submissions/submission_latest.csv` | Most recent submission           |
| `artifacts/experiments/<timestamp>/`          | Full experiment snapshot         |
