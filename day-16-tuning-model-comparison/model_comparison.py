"""Practical task (Module A): Compare multiple models and justify the best one.

Pipeline: scale -> tune each model with GridSearchCV -> compare on CV metrics
and on a held-out test set -> pick a winner.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# (estimator, hyperparameter grid) for each candidate
candidates = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7, 11]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {"model__max_depth": [3, 5, 7, None]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"model__n_estimators": [100, 300], "model__max_depth": [None, 5, 10]}),
    "SVM (RBF)": (SVC(probability=True, random_state=42),
                  {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]}),
}

rows = []
for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV F1": search.best_score_,
        "Test Precision": precision_score(y_test, pred),
        "Test Recall": recall_score(y_test, pred),
        "Test F1": f1_score(y_test, pred),
        "Test ROC-AUC": roc_auc_score(y_test, proba),
        "Best params": search.best_params_,
    })

results = pd.DataFrame(rows).sort_values("CV F1", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 200)
print(results.drop(columns="Best params").round(4).to_string(index=False))
print("\nBest params per model:")
for _, r in results.iterrows():
    print(f"  {r['Model']:<20} {r['Best params']}")

winner = results.iloc[0]
print(f"\nWinner: {winner['Model']} (CV F1={winner['CV F1']:.4f}, "
      f"Test F1={winner['Test F1']:.4f}, Test recall={winner['Test Recall']:.4f})")

results.drop(columns="Best params").round(4).to_csv("outputs/model_comparison_results.csv", index=False)

# Bar chart of CV F1 vs test F1
ax = results.set_index("Model")[["CV F1", "Test F1"]].plot.bar(figsize=(8, 4.5), rot=20)
ax.set_ylim(0.9, 1.0)
ax.set_ylabel("F1-score")
ax.set_title("Model comparison (tuned with GridSearchCV)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
print("Saved outputs/model_comparison.png and model_comparison_results.csv")
