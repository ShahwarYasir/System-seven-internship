"""Q13 (Hard): Build a Pipeline (scaler -> PCA -> classifier), then show data
leakage by scaling before the split and compare scores.

Part A builds the pipeline.
Part B scales before the split (leaky) vs inside a pipeline (correct), over many
        random splits, as the question asks.
Part C shows a dramatic case: feature selection before the split on pure noise.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

# ---------------- Part A: the pipeline -------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression(max_iter=1000)),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_train, y_train)
print("PART A - Pipeline (scaler -> PCA -> logistic regression)")
print(f"  Test accuracy: {pipe.score(X_test, y_test):.4f}")
print(f"  Variance kept by 10 components: {pipe.named_steps['pca'].explained_variance_ratio_.sum()*100:.2f}%\n")

# ---------------- Part B: scaling leakage ----------------------------------
leaky, correct = [], []
for seed in range(50):
    # WRONG: the scaler sees the whole dataset, including future test rows
    X_all_scaled = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_all_scaled, y, test_size=0.2,
                                          random_state=seed, stratify=y)
    leaky.append(Pipeline([("pca", PCA(n_components=10)),
                           ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr).score(Xte, yte))
    # RIGHT: split first, the scaler learns from the training part only
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=seed, stratify=y)
    correct.append(pipe.fit(Xtr, ytr).score(Xte, yte))

print("PART B - Scaling before the split (50 random splits)")
print(f"  Leaky   (scale -> split): mean accuracy = {np.mean(leaky):.4f}")
print(f"  Correct (split -> scale): mean accuracy = {np.mean(correct):.4f}")
print(f"  Difference: {(np.mean(leaky) - np.mean(correct))*100:+.2f} percentage points")
print("  (scaling leakage is usually small, because only the mean and std leak)\n")

# ---------------- Part C: feature-selection leakage on pure noise ----------
rng = np.random.default_rng(0)
X_noise = rng.normal(size=(100, 5000))      # 5000 random features
y_noise = rng.integers(0, 2, size=100)       # random labels: nothing to learn

# WRONG: pick the 20 "best" features using ALL labels, then cross-validate
X_sel = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
leaky_cv = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_noise, cv=5).mean()

# RIGHT: feature selection lives inside the pipeline, refitted in every fold
safe = Pipeline([("select", SelectKBest(f_classif, k=20)),
                 ("clf", LogisticRegression(max_iter=1000))])
safe_cv = cross_val_score(safe, X_noise, y_noise, cv=5).mean()

print("PART C - Feature selection on random noise (true accuracy should be ~0.50)")
print(f"  Leaky   (select -> CV): accuracy = {leaky_cv:.4f}")
print(f"  Correct (pipeline CV) : accuracy = {safe_cv:.4f}")
