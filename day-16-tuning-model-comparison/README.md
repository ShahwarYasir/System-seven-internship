# Day 16 - Tuning and Model Comparison

**System 7 | Week 3 | Module A (Day 2)**
Topics: over/underfitting, feature scaling, encoding, GridSearchCV, RandomizedSearchCV, model comparison task.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q6_overfitting_decision_tree.py` | Q6 | Medium | Train vs test accuracy for `max_depth` 1 to 15 |
| `q7_column_transformer.py` | Q7 | Medium | Scale numeric + one-hot encode categorical columns with `ColumnTransformer` |
| `q8_gridsearch_vs_randomsearch.py` | Q8 | Medium | Tune an SVM with both searches; compare score and time |
| `model_comparison.py` | Practical task | | Tune 5 models and justify the best one |
| `outputs/` | | | Plots and the comparison CSV |

## How to run

```bash
pip install -r requirements.txt
mkdir -p outputs
python q6_overfitting_decision_tree.py   # and so on
python model_comparison.py
```

## Results

**Q6.** Training accuracy climbs to 1.0 by depth 6, while test accuracy peaks at depth 5 (0.9298) and then drops to about 0.918. The widening gap is overfitting. Plot: `outputs/q6_overfitting_curve.png`.

**Q7.** `ColumnTransformer` turns 5 raw columns into 9 model-ready features (3 scaled numeric plus 6 one-hot columns). It is fitted on training data only, then reused on test data, and it plugs directly into a `Pipeline`.

**Q8.** Both searches reach the same best CV score (0.9802) and test score (0.9825), but random search tries 20 candidates instead of 75 and finishes about 4x faster.

| | Candidates | Best CV | Test | Time |
|---|---|---|---|---|
| GridSearchCV | 75 | 0.9802 | 0.9825 | about 2.1 s |
| RandomizedSearchCV | 20 | 0.9802 | 0.9825 | about 0.5 s |

Timings vary by machine.

### Practical task: model comparison

Five models, each scaled and tuned with `GridSearchCV` (5-fold, scored on F1), then evaluated on a held-out test set:

| Model | CV F1 | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9861** | 0.9726 | 0.9861 | 0.9793 | 0.9957 |
| SVM (RBF) | 0.9810 | 0.9861 | 0.9861 | 0.9861 | 0.9977 |
| Random Forest | 0.9718 | 0.9583 | 0.9583 | 0.9583 | 0.9937 |
| KNN | 0.9708 | 0.9600 | 1.0000 | 0.9796 | 0.9911 |
| Decision Tree | 0.9400 | 0.9452 | 0.9583 | 0.9517 | 0.9446 |

**Best model: Logistic Regression.** It has the highest cross-validated F1, which is a more reliable signal than a single 114-row test split. Its test recall (0.986) is among the best, which matters because a missed malignant case is the costly error. It is also the simplest and most interpretable model, and the SVM is only marginally better on one test split, which is within noise. Plots: `outputs/model_comparison.png`, `outputs/model_comparison_results.csv`.

## Learning note: bias-variance trade-off

- **High bias (underfitting):** the model is too simple to capture the pattern, so it does badly on both training and test data. Example: `max_depth=1` in Q6.
- **High variance (overfitting):** the model is so flexible it memorises noise, so training score is excellent but test score is worse. Example: `max_depth` above 6 in Q6, with training accuracy 1.0 and test accuracy falling.
- **The sweet spot** is the complexity where test error is lowest (depth 5 in Q6).

Analogy: studying for an exam. Bias is skimming only the chapter titles, so you fail everything. Variance is memorising last year's exact questions, so you ace those and fail any new question. Good generalization is understanding the concepts.

Ways to fix overfitting: simpler model, more data, regularisation, cross-validation, limiting depth.
