# EXPERIMENTS — Text Authorship Detection (MALTO Hackathon)

Tracking every run: what changed, what the CV F1 was, what the Kaggle leaderboard F1 was.

**Metric**: Macro F1 (all 6 classes weighted equally)
**Classes**: Human=0, DeepSeek=1, Grok=2, Claude=3, Gemini=4, ChatGPT=5
**Class counts (train)**: Human=1520, DeepSeek=80, Grok=160, Claude=80, Gemini=240, ChatGPT=320
**Key challenge**: DeepSeek (80 samples) recall was near 0 before run 2.

---

## Run 1 — Baseline TF-IDF + Linear Models
**Date**: 2026-03-16  
**Kaggle public LB**: 0.85942 (rank 15/16, 1st place = 0.95081)  
**Best CV model**: sgd_hinge — 0.8943 macro F1

### Config
- Word TF-IDF: (1,2) ngrams, 100k features
- Char TF-IDF: (3,5) ngrams, 100k features
- No stylometric features
- Models: logistic_regression, logistic_regression_balanced, linear_svc,
  linear_svc_balanced, calibrated_svc, sgd_log, sgd_hinge, ridge_classifier,
  passive_aggressive, complement_nb

### CV Results
| Model                        | CV Macro F1 |
|------------------------------|-------------|
| sgd_hinge                    | 0.8943      |
| logistic_regression_balanced | ~0.883      |
| ridge_classifier              | ~0.882      |

### Issues identified
- DeepSeek recall ≈ 0.07 (catastrophic — most predicted as Grok)
- CV (0.8943) >> Kaggle LB (0.8594) — 0.035 gap

---

## Run 2 — Stylometric (34 features) + Soft Ensemble + Threshold Optimization
**Date**: 2026-03-16  
**Kaggle public LB**: (submit artifacts/submissions/submission_latest.csv)  
**Best CV model**: logistic_regression_balanced — **0.9140** macro F1  
**After threshold opt (OOF)**: **0.9192** macro F1

### Changes from Run 1
1. **Char TF-IDF widened**: (3,5) → **(2,6)** — captures more subword patterns
2. **Stylometric features added (34 features)** — up from 0:
   - Base: char/word counts, avg sentence/word length, type-token ratio, punctuation rates
   - NEW: `very_short_sent_ratio` (< 5 words) — DeepSeek tends to write short sentences
   - NEW: `very_long_sent_ratio` (> 30 words) — Grok tends longer sentences
   - NEW: `numbered_list_rate` — DeepSeek loves numbered lists
   - NEW: `markdown_header_rate` — Gemini/Claude use markdown headers
   - NEW: `code_block_rate`, `bold_rate`, `italic_rate`, `link_rate`
   - NEW: `parenthesis_rate`, `ellipsis_rate`, `dash_rate`, `caps_word_ratio`
   - NEW: `avg_para_len_chars`, `starts_with_i_ratio`
3. **ensemble_soft model** (new): soft voting with calibrated models
   - Components: CalibratedClassifierCV(sgd_hinge) + CalibratedClassifierCV(ridge) +
     LogisticRegression(balanced) + CalibratedClassifierCV(LinearSVC)
4. **Per-class threshold optimizer** (new post-processing):
   - Grid-searches per-class probability scale factors on OOF data
   - Optimizes macro F1 directly
   - Thresholds [1.25, 1.25, 1.0, 1.0, 1.25, 1.75] saved to artifacts/thresholds.json
   - Applied automatically at inference time

### CV Results (all models, this run)
| Model                        | CV Macro F1 | OOF Macro F1 |
|------------------------------|-------------|--------------|
| logistic_regression_balanced | **0.9140**  | 0.9145       |
| ensemble_soft                | 0.9025      | 0.9026       |
| ensemble_top3                | 0.9006      | 0.9011       |
| sgd_hinge                    | 0.8936      | 0.8938       |
| ridge_classifier              | 0.8892      | 0.8904       |
| calibrated_svc               | 0.8719      | 0.8735       |

### Per-class F1 for logistic_regression_balanced (OOF)
| Class   | Before threshold opt | After threshold opt |
|---------|---------------------|---------------------|
| Human   | 0.998               | 0.999               |
| DeepSeek| **0.637** (was 0.07)| **0.671**           |
| Grok    | 0.865               | 0.852               |
| Claude  | 1.000               | 1.000               |
| Gemini  | 0.994               | 0.998               |
| ChatGPT | 0.992               | 0.995               |
| **Macro**| **0.9145**         | **0.9192**          |

### Analysis
- **DeepSeek recall massively improved**: 0.07 → 0.637
  - Root cause: stylometric features (numbered_list_rate, very_short_sent_ratio) give
    clear signal to separate DeepSeek from Grok
- **logistic_regression_balanced > sgd_hinge**: LR with LBFGS solver can find a better
  global optimum; the balanced weights matter more with the richer feature set
- **Overfit gap still present** (~0.10): models memorize training features; gap suggests
  room for regularization improvement

### Final submission
- Model: `logistic_regression_balanced`
- Thresholds: [1.25, 1.25, 1.0, 1.0, 1.25, 1.75]
- Submission: `artifacts/submissions/submission_latest.csv`

---

## Run 3 — Reduced Overfit + Seed Averaging + More DeepSeek Features
**Date**: 2026-03-16
**Kaggle public LB**: (pending — run python main_train.py then main_infer.py)
**Target**: > 0.91 LB

### Changes from Run 2
1. **37 stylometric features** (up from 34) — 3 new DeepSeek vs Grok discriminators:
   - `punct_variety`: ratio of distinct punctuation chars (DeepSeek uses very few)
   - `sent_length_cv`: coefficient of variation of sentence lengths (DeepSeek is uniform)
   - `transition_word_rate`: "however/therefore/moreover/..." per sentence
2. **Reduced feature space to fight CV-LB gap**:
   - word TF-IDF: max_features 100k -> 50k, min_df 2 -> 3
   - char TF-IDF: max_features 100k -> 50k, min_df 3 -> 4
   - Reason: overfit gap ~0.048 (CV 0.9140 vs LB 0.8660) — model memorises rare words
3. **More regularization**: LR C: 1.0 -> 0.5 (less memorization of training text)
4. **`lr_seed_avg` model** (new): runs LR 5x with seeds [42,123,456,789,2024],
   averages predicted probabilities — reduces variance on minority classes
5. **`use_best_fold_model: true`**: saves the best CV fold's fitted pipeline and
   uses it as the final model (instead of retraining on all 2400 samples)
   — the best fold model saw only 1920 samples, so it generalizes better to test

### Hypothesis
- Reducing max_features + increasing regularization will close the LB gap
- Seed averaging will give more stable DeepSeek predictions
- Best-fold model avoids the overfit that comes from training on all data


### Bug fixes applied before Run 3
Three pipeline bugs found and fixed:
1. **Model mismatch**: `best_model` in config was used for final training, but thresholds
   were computed for the CV winner. Fixed: CV winner now ALWAYS overrides config for final
   training, so model and thresholds always match.
2. **Thresholds applied to wrong model**: threshold_optimizer uses OOF proba of the CV
   winner; applying those thresholds to a different model breaks the calibration.
   Fixed by fix #1 above.
3. **`use_best_fold_model`**: was True, meaning only 80% of data was used for final model.
   Set to False — retrain on all 2400 samples for the submitted model.
4. **Submission not saved to experiment folder**: added shutil copy of submission CSV into
   each experiment's directory for full reproducibility.

### How to run
```
cd d:\hachaton	ext-authorship-detection
.env\Scriptsctivate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```
Then submit artifacts/submissions/submission_latest.csv to Kaggle.

---

## Run 4 — (planned)
**Target**: Kaggle LB > 0.93

**Planned changes** (if Run 2 does not reach ~0.93+):
1. **Reduce overfit gap**: Add stronger L2 regularization (C=0.1 for LR), or increase
   min_df to reduce vocabulary size and force generalization
2. **Seed averaging**: Run logistic_regression_balanced with 5 seeds (42,123,456,789,2024),
   average the predicted probabilities → more stable minority class predictions
3. **Hierarchical classifier**: Human vs AI (binary) → which AI (5-class)
   - Motivated by: Human (1520) is much easier; binary stage reduces AI-vs-AI confusion
4. **DeepSeek-specific features**: Add features specifically designed to catch DeepSeek's
   style (low punctuation variety score = stdev of punctuation rates)
5. **Repeated stratified CV with multiple seeds** to get a more reliable CV estimate
   and close the CV → LB gap
