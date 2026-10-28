"""Q4 (Medium): 5-fold cross-validation on three models (mean and std)."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# Stratified folds keep the class ratio in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Model':<22}{'Mean acc':>10}{'Std':>10}")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:<22}{scores.mean():>10.4f}{scores.std():>10.4f}   folds={scores.round(3)}")
