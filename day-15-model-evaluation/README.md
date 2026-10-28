# Day 15 - Model Evaluation

**System 7 | Week 3 | Module A (Day 1)**
Topics: train-test split, cross-validation, classification and regression metrics, confusion matrix, ROC-AUC.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q1_train_test_split.py` | Q1 | Easy | 80/20 stratified split, prints shapes and class balance |
| `q2_classification_metrics.py` | Q2 | Easy | Accuracy, precision, recall, F1 and confusion-matrix plot |
| `q3_regression_metrics.py` | Q3 | Easy | MAE, MSE, RMSE, R2 in NumPy, verified with scikit-learn |
| `q4_cross_validation.py` | Q4 | Medium | 5-fold CV on three models, mean and std |
| `q5_roc_auc.py` | Q5 | Medium | ROC curves and AUC for logistic regression vs random forest |
| `outputs/` | | | Saved plots |

## How to run

```bash
pip install -r requirements.txt
mkdir -p outputs
python q1_train_test_split.py   # and so on for q2..q5
```

## Results

**Q1.** With `stratify=y` and `random_state=42`, the split is 455 train / 114 test rows. Class balance stays close to the original (benign about 63%, malignant about 37%) in both parts.

**Q2.** Logistic regression on the breast cancer dataset: accuracy 0.9825, precision 0.9861, recall 0.9861, F1 0.9861. Plot: `outputs/q2_confusion_matrix.png`.

**Q3.** The hand-written NumPy metrics match scikit-learn exactly (MAE about 24,155, RMSE about 29,398, R2 about 0.934) on a synthetic house-price dataset.

**Q4.** 5-fold stratified CV (accuracy):

| Model | Mean | Std |
|---|---|---|
| Logistic Regression | 0.9737 | 0.0166 |
| KNN (k=5) | 0.9631 | 0.0179 |
| Random Forest | 0.9543 | 0.0102 |

**Q5.** AUC: logistic regression 0.9954, random forest 0.9931. Plot: `outputs/q5_roc_curves.png`.

## Learning note: choosing the right metric

Accuracy only counts how often the model is right, so it can look great while being useless. If 99 of 100 emails are not spam, a model that always says "not spam" scores 99% and catches nothing.

- **Precision** asks: of everything I flagged, how much was truly positive? It matters when a false alarm is costly (for example, blocking a legitimate payment).
- **Recall** asks: of all the real positives, how many did I catch? It matters when a miss is costly (for example, missing a cancer case or a fraud attempt).
- **F1** is the harmonic mean of the two, useful when you need a balance and the classes are imbalanced.
- **ROC-AUC** measures how well the model ranks positives above negatives across all thresholds, independent of any single cutoff.

Analogy: a fishing net. Recall is how much of the fish you caught, precision is how much of the catch is fish rather than boots. A huge net catches every fish (high recall) but also a lot of boots (low precision).

In Q2, the cancer dataset is exactly the case where recall matters: a missed malignant tumour is worse than a false alarm.
