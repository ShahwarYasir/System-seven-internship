"""Q5 (Medium): ROC curves and AUC for logistic regression vs random forest."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

fig, ax = plt.subplots(figsize=(6, 5))
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]   # probability of the positive class
    auc = roc_auc_score(y_test, proba)
    print(f"{name:<20} AUC = {auc:.4f}")
    RocCurveDisplay.from_predictions(y_test, proba, name=f"{name} (AUC={auc:.3f})", ax=ax)

ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_title("ROC curves")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/q5_roc_curves.png", dpi=150)
print("Saved outputs/q5_roc_curves.png")
