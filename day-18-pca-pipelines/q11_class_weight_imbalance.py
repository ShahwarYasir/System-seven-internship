"""Q11 (Medium): Handle an imbalanced dataset with class_weight and compare
recall before and after."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 95% negative, 5% positive (for example fraud or a rare disease)
X, y = make_classification(
    n_samples=5000, n_features=12, n_informative=6, n_redundant=2,
    weights=[0.95, 0.05], flip_y=0.01, class_sep=0.8, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Class balance (train):", pd.Series(y_train).value_counts().sort_index().to_dict())

def build(kind, weight):
    if kind == "Logistic Regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight=weight))
    return RandomForestClassifier(n_estimators=200, class_weight=weight, random_state=42)

rows, preds = [], {}
for kind in ["Logistic Regression", "Random Forest"]:
    for weight in [None, "balanced"]:
        model = build(kind, weight).fit(X_train, y_train)
        pred = model.predict(X_test)
        label = f"{kind} | class_weight={weight}"
        preds[label] = pred
        rows.append({
            "Model": kind,
            "class_weight": str(weight),
            "Accuracy": (pred == y_test).mean(),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })

results = pd.DataFrame(rows).round(4)
print("\n" + results.to_string(index=False))

# Confusion matrices for logistic regression before vs after
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, label in zip(axes, ["Logistic Regression | class_weight=None",
                            "Logistic Regression | class_weight=balanced"]):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[label], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(label.split("|")[1].strip())
plt.tight_layout()
plt.savefig("outputs/q11_class_weight_confusion.png", dpi=150)
print("\nSaved outputs/q11_class_weight_confusion.png")
