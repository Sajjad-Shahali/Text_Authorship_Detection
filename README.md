# MALTO Recruitment Hackathon 2026 — Text Authorship Detection

**Author:** [Sajjad Shahali](https://github.com/Sajjad-Shahali)
**GitHub:** [text-authorship-detection](https://github.com/Sajjad-Shahali/text-authorship-detection)
**Competition:** MALTO Recruitment Hackathon (2026)
**Task:** 6-class multiclass text authorship classification
**Metric:** Macro F1 Score (all classes weighted equally)
**Constraint:** No external data · No pre-trained AI detectors · No transformers

| Score | Value |
|-------|-------|
| **Best Public Leaderboard** | **0.92422** |
| **Best Local CV (5-fold)** | **0.9393** |
| **Best Competitor** | 0.96423 |
| **Gap to Leader** | 0.04001 |

---

## Problem Statement

Given a raw text sample, classify which of 6 possible sources produced it:

| Label | Source | Train Samples | Class % |
|-------|--------|:---:|:---:|
| 0 | Human-written | 1520 | 63.3% |
| 1 | DeepSeek | 80 | 3.3% |
| 2 | Grok | 160 | 6.7% |
| 3 | Claude | 80 | 3.3% |
| 4 | Gemini | 240 | 10.0% |
| 5 | ChatGPT | 320 | 13.3% |

**Total:** 2,400 train · 600 test
**Core challenge:** DeepSeek (80 samples) and Grok (160 samples) both produce short, factual,
encyclopedia-style text. Distinguishing them is the primary bottleneck.

---

## Best Model: `stacking_lgbm` (Run 10, LB 0.92422)

### Pipeline Architecture

```
Raw text
  └─► Preprocessor  (unicode normalization, whitespace — preserve style signals)
        └─► FeatureUnion  (~100k+ total dimensions)
              ├─ Word TF-IDF        [50k features, ngram (1,2), word]
              ├─ Char TF-IDF        [50k features, ngram (2,6), char_wb]
              ├─ Stylometric        [43 hand-crafted features]
              └─ Function-word TF-IDF [151 fixed vocabulary words]
                    └─► StackingClassifier (3-fold internal CV)
                          ├─ Base 1: TwoStageClassifier
                          │    ├─ LogisticRegression(C=0.5, balanced) — 6-class
                          │    └─ LogisticRegression(C=1.5, balanced) — DS/Grok binary
                          ├─ Base 2: TfidfMLPClassifier
                          │    └─ TruncatedSVD(500) → MLP(512→256, ReLU, adam)
                          └─ Base 3: LGBMTfidfClassifier
                               └─ TruncatedSVD(300) → LightGBM(300 trees, 31 leaves)
                                    └─► Meta: LogisticRegression(C=0.1, balanced)
                                          └─► Per-class threshold scaling
                                                └─► DS/Grok pair ratio threshold
```

### Feature Engineering Details

#### 1. Word TF-IDF
- `analyzer="word"`, `ngram_range=(1,2)`, `max_features=50000`, `min_df=3`, `sublinear_tf=True`
- Captures vocabulary and phrase-level style; `min_df=3` prevents overfitting on hapax words

#### 2. Character TF-IDF
- `analyzer="char_wb"`, `ngram_range=(2,6)`, `max_features=50000`, `min_df=2`, `sublinear_tf=True`
- Word-boundary character n-grams capture punctuation, spacing, morphological patterns
- Extremely discriminative for LLM fingerprinting

#### 3. Stylometric Features (43 dimensions)

| Group | Features | Key signal |
|-------|----------|-----------|
| Length | char count, word count, sentence count, para count | Text volume |
| Lexical | type-token ratio, avg word length, long-word ratio | Vocabulary diversity |
| Sentence shape | avg len, very-short/long ratios, sent range | DS uniform short, Grok high-variance |
| Punctuation | comma, period, colon, semicolon, quote, paren, ellipsis, dash | Per-word rates |
| Char composition | uppercase ratio, digit ratio, caps-word ratio, newline ratio | DS vs Gemini |
| Markdown | bullets, headers, numbered lists, code, bold, italic, links | Claude/Gemini use headers |
| Discourse | transition words, hedging language, starts-with-I ratio | Academic vs conversational |
| DS/Grok discriminators | punct variety, sent CV, first-sent words, proper noun density, log-length | Fine-grained |

#### 4. Function-Word TF-IDF
- Fixed 151-word vocabulary: determiners, prepositions, conjunctions, auxiliaries, pronouns, discourse markers
- Topic-neutral style fingerprint — captures *how* the author connects ideas, not *what* they write

### Post-Processing: Threshold Optimization

**Step 1 — Per-class probability scaling:**

$$\hat{y} = \arg\max_{k} \; ( p_k \cdot s_k )$$

Run 10 optimal scales: `[2.25, 0.75, 1.0, 1.0, 2.5, 1.75]`

**Step 2 — DeepSeek/Grok pair ratio threshold:**

$$\hat{y} = \begin{cases} \text{DeepSeek} & \text{if } \frac{P_{\text{DS}}}{P_{\text{DS}} + P_{\text{Grok}}} \geq \tau \\ \text{Grok} & \text{otherwise} \end{cases}$$

Run 10 optimal $\tau = 0.55$

### Stacking Architecture

The `StackingClassifier` uses 3-fold internal CV to generate 18-dimensional meta-features
(3 models × 6 class probabilities) for the 2,400 training samples. The meta
LogisticRegression (C=0.1, strongly regularized) learns optimal trust weights per model per class.

The 3 base estimators are deliberately **orthogonal**:

| Model | Representation | Boundary | Strength |
|-------|---------------|----------|----------|
| TwoStageClassifier | Sparse TF-IDF | Linear + binary | Dedicated DS/Grok specialist |
| TfidfMLPClassifier | Dense SVD(500) | Non-linear (MLP) | Feature interaction modeling |
| LGBMTfidfClassifier | Dense SVD(300) | Non-linear (trees) | Robust, interpretable |

---

## Performance History

| Run | Best Model | CV F1 | LB Score | Key Change |
|-----|------------|------:|--------:|------------|
| 1 | sgd_hinge | 0.8943 | 0.85942 | Baseline TF-IDF + linear models |
| 2 | lr_balanced | 0.9140 | — | 34 stylometric features + threshold optimization |
| 3 | lr_balanced | 0.9140 | — | Regularization, vocabulary pruning |
| 4 | two_stage | 0.9328 | 0.91089 | DS/Grok binary stage + function-word TF-IDF |
| 5 | two_stage_top2 | 0.9328 | 0.91089 | Top-2 trigger condition |
| 6 | two_stage_top2 | 0.9328 | 0.91089 | 43 stylometric features (expanded from 34) |
| 7 | two_stage_top2 | 0.9328 | 0.91089 | Threshold variants (no improvement) |
| 8 | ensemble_mlp | 0.9350 | 0.90978 | MLP + TruncatedSVD (CV up, LB down) |
| 9 | ensemble_mlp | 0.9350 | — | MLP calibration experiments |
| **10** | **stacking_lgbm** | **0.9393** | **0.92422** | **Stacking: TwoStage + MLP + LGBM** |
| 11 | stacking_lgbm | 0.9370 | — | Bug fixes, stacking v2 experiments |
| 12 | stacking_lgbm | 0.9339 | — | New feature branches (regression) |

### Run 10 OOF Per-class Performance

```
              precision    recall  f1-score   support

       Human       1.00      1.00      1.00      1520
    DeepSeek       0.76      0.78      0.77        80
        Grok       0.89      0.90      0.90       160
      Claude       1.00      1.00      1.00        80
      Gemini       0.99      0.98      0.99       240
     ChatGPT       1.00      0.99      0.99       320

    macro avg       0.94      0.94      0.94      2400
```

DeepSeek (77 F1) and Grok (90 F1) remain the bottleneck — 32 total confusion pair errors.

---

## Repository Structure

```
text-authorship-detection/
├── configs/
│   └── config.yaml              # All hyperparameters (see configs/README.md)
│
├── src/
│   ├── constants.py             # Label maps, column names
│   ├── utils.py                 # Logging, YAML config, artifact helpers
│   ├── data.py                  # CSV loading + schema validation
│   ├── preprocess.py            # Minimal configurable text cleaning
│   ├── features.py              # TF-IDF + stylometric FeatureUnion
│   ├── models.py                # Model factory: all classifiers
│   ├── train.py                 # CV loop + final training (anti-leakage)
│   ├── evaluate.py              # Macro F1, classification report
│   ├── inference.py             # Load pipeline, predict on test data
│   ├── threshold_optimizer.py   # Per-class & DS/Grok threshold search
│   └── submission.py            # Format + validate Kaggle submission
│   (see src/README.md)
│
├── artifacts/
│   ├── experiments/
│   │   └── 2026-03-16_221821_run/   # Best run: LB 0.92422
│   │       ├── best_model.joblib    # Fitted pipeline (661 MB)
│   │       ├── thresholds.json      # Optimized thresholds
│   │       └── config_snapshot.yaml # Frozen config (reproducibility)
│   └── submissions/                 # All submission CSVs
│   (see artifacts/README.md)
│
├── data/
│   ├── train.csv                # 2400 labeled samples
│   ├── test.csv                 # 600 unlabeled samples
│   └── sample_submission.csv   # Format reference
│   (see data/README.md)
│
├── notebooks/
│   └── submission_notebook.ipynb  # Complete reproducible training + submission
│   (see notebooks/README.md)
│
├── main_train.py                # Entry point: full training pipeline
├── main_infer.py                # Entry point: inference on test set
├── requirements.txt
├── Makefile
├── EXPERIMENTS.md               # Detailed run-by-run experiment log
└── HANDOFF.md                   # Current project status + next steps
```

---

## Setup

### Requirements

```
Python >= 3.9
scikit-learn >= 1.3.0
lightgbm >= 4.0.0
numpy >= 1.24.0
pandas >= 2.0.0
joblib >= 1.3.0
PyYAML >= 6.0
scipy >= 1.10.0
```

### 1. Clone

```bash
git clone https://github.com/Sajjad-Shahali/text-authorship-detection.git
cd text-authorship-detection
```

### 2. Environment

```bash
python -m venv .env

# Windows
.env\Scripts\activate

# Linux / macOS
source .env/bin/activate

pip install -r requirements.txt
```

### 3. Data

Place the competition files in `data/`:

```
data/train.csv
data/test.csv
data/sample_submission.csv
```

### 4. Train

```bash
python main_train.py --config configs/config.yaml
# or:
make train
```

This will:
1. Create a timestamped experiment folder under `artifacts/experiments/`
2. Run 5-fold stratified cross-validation for all configured models
3. Select the CV winner and train on the full 2,400-sample dataset
4. Optimize per-class thresholds on out-of-fold predictions
5. Save `best_model.joblib`, `thresholds.json`, `config_snapshot.yaml`, and all metrics

### 5. Inference

```bash
python main_infer.py --config configs/config.yaml
# or:
make infer
```

Produces: `artifacts/submissions/submission_<timestamp>.csv`

### 6. Tests

```bash
make test          # 33 unit tests, no data files required
make test-cov      # with coverage report
```

---

## Reproducibility (Run 10, LB 0.92422)

Use the frozen config snapshot:

```bash
cp artifacts/experiments/2026-03-16_221821_run/config_snapshot.yaml configs/config.yaml
python main_train.py --config configs/config.yaml
```

Key Run 10 settings:

```yaml
features:
  word_tfidf:    {ngram_range: [1,2], max_features: 50000, min_df: 3}
  char_tfidf:    {ngram_range: [2,6], max_features: 50000, min_df: 2, analyzer: char_wb}
  stylometric:   {enabled: true}      # 43 features
  function_word_tfidf: {enabled: true}

models:
  run_models: [stacking_lgbm]
  lgbm: {n_svd_components: 300, n_estimators: 300, num_leaves: 31, learning_rate: 0.05}
  mlp:  {n_svd_components: 500, hidden_layer_sizes: [512, 256], alpha: 0.01}
```

Frozen thresholds:
```json
{"thresholds": [2.25, 0.75, 1.0, 1.0, 2.5, 1.75], "ds_grok_pair_threshold": 0.55}
```

### Kaggle Notebook

`notebooks/submission_notebook.ipynb` is a self-contained notebook that:
- Defines all custom classes inline (no `src/` required)
- Trains the full `stacking_lgbm` pipeline from scratch
- Runs 5-fold CV + threshold optimization
- Generates a timestamped submission CSV
- Auto-detects Kaggle vs local environment

---

## Anti-Leakage Design

The full pipeline (Preprocessor → FeatureUnion → Classifier) is **rebuilt and fitted inside
each CV fold**. Vectorizer vocabulary, IDF weights, stylometric scaler, and classifier weights
are computed only on training-fold data. No information from validation samples influences
feature statistics.

---

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| No lowercasing | Capitalization is a style signal (Claude uses **bold**, Gemini uses headers) |
| Preserve punctuation | Comma/semicolon rates distinguish LLM writing styles |
| `min_df=3` for word TF-IDF | Hapax words don't generalize — reduces train/test gap |
| `char_wb` analyzer | Word-boundary char n-grams capture space+word patterns |
| Balanced class weights | DeepSeek (3.3%) would be invisible without reweighting |
| Stacking over voting | Meta-LR learns per-model confidence, not just averaging |
| Per-class thresholds | Corrects calibration errors on minority classes |
| DS/Grok pair threshold | Ratio threshold more stable than absolute probability |

---

## Environment

```
Python  3.12.10
scikit-learn  1.8.0
lightgbm  4.6.0
numpy  2.4.3
pandas  3.0.1
scipy  1.17.1
joblib  1.5.3
Platform: Windows 11 Pro
```

---

## License

MIT License

---

*Built for the MALTO Recruitment Hackathon 2026*
*[Sajjad Shahali](https://github.com/Sajjad-Shahali) · [text-authorship-detection](https://github.com/Sajjad-Shahali/text-authorship-detection)*
