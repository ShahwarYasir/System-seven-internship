"""Q7 (Medium): Scale numeric columns and one-hot encode categorical columns
in a single step with ColumnTransformer."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Small synthetic student dataset with mixed column types ---------------
rng = np.random.default_rng(42)
n = 400
df = pd.DataFrame({
    "study_hours": rng.uniform(0, 10, n),
    "attendance": rng.uniform(40, 100, n),
    "prev_gpa": rng.uniform(1.5, 4.0, n),
    "gender": rng.choice(["male", "female"], n),
    "program": rng.choice(["AI", "CS", "SE", "DS"], n),
})
score = 0.4 * df.study_hours + 0.03 * df.attendance + 0.8 * df.prev_gpa + rng.normal(0, 0.7, n)
df["passed"] = (score > score.median()).astype(int)

num_cols = ["study_hours", "attendance", "prev_gpa"]
cat_cols = ["gender", "program"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X, y = df[num_cols + cat_cols], df["passed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# fit on training data only, then reuse on test data
X_train_t = preprocess.fit_transform(X_train)
X_test_t = preprocess.transform(X_test)

print("Original train shape   :", X_train.shape)
print("Transformed train shape:", X_train_t.shape)
print("Output feature names   :", list(preprocess.get_feature_names_out()))
print("\nScaled numeric means (should be ~0):",
      np.round(X_train_t[:, :3].mean(axis=0), 3))

# The transformer plugs straight into a model
clf = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
clf.fit(X_train, y_train)
print(f"\nPipeline test accuracy: {clf.score(X_test, y_test):.4f}")
