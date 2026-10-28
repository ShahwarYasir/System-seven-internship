"""Q8 (Medium): Tune an SVM with GridSearchCV and RandomizedSearchCV and
compare best score and run time."""
import time
import pandas as pd
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([("scale", StandardScaler()), ("svm", SVC())])

# --- Grid search: tries every combination (5 x 5 x 3 = 75) ------------------
param_grid = {
    "svm__C": [0.1, 1, 10, 100, 1000],
    "svm__gamma": [0.0001, 0.001, 0.01, 0.1, 1],
    "svm__kernel": ["rbf", "linear", "poly"],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
t0 = time.perf_counter()
grid.fit(X_train, y_train)
grid_time = time.perf_counter() - t0

# --- Random search: samples 20 combinations from continuous ranges ----------
param_dist = {
    "svm__C": loguniform(1e-1, 1e3),
    "svm__gamma": loguniform(1e-4, 1e0),
    "svm__kernel": ["rbf", "linear", "poly"],
}
rand = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=5, scoring="accuracy",
                          n_jobs=-1, random_state=42)
t0 = time.perf_counter()
rand.fit(X_train, y_train)
rand_time = time.perf_counter() - t0

summary = pd.DataFrame({
    "Candidates tried": [len(grid.cv_results_["params"]), len(rand.cv_results_["params"])],
    "Best CV score": [grid.best_score_, rand.best_score_],
    "Test score": [grid.score(X_test, y_test), rand.score(X_test, y_test)],
    "Time (s)": [grid_time, rand_time],
}, index=["GridSearchCV", "RandomizedSearchCV"]).round(4)

print(summary, "\n")
print("Grid best params  :", grid.best_params_)
print("Random best params:", rand.best_params_)
