"""Q3 (Easy): MAE, MSE, RMSE and R2 for a house-price model.

Metrics are computed by hand with NumPy, then verified against scikit-learn.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split

# --- Synthetic house-price data (area, bedrooms, age -> price) -------------
rng = np.random.default_rng(42)
n = 500
area = rng.uniform(50, 300, n)          # square metres
bedrooms = rng.integers(1, 6, n)
age = rng.uniform(0, 40, n)             # years
price = 1500 * area + 8000 * bedrooms - 600 * age + rng.normal(0, 25000, n)
X = np.column_stack([area, bedrooms, age])

X_train, X_test, y_train, y_test = train_test_split(
    X, price, test_size=0.2, random_state=42
)
y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)

# --- Manual NumPy implementations -----------------------------------------
errors = y_test - y_pred
mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum(errors ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

# --- scikit-learn versions ------------------------------------------------
sk = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R2": r2_score(y_test, y_pred),
}
manual = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

print(f"{'Metric':<6} {'NumPy':>16} {'scikit-learn':>16}  match")
for k in manual:
    ok = np.isclose(manual[k], sk[k])
    print(f"{k:<6} {manual[k]:>16.4f} {sk[k]:>16.4f}  {ok}")
