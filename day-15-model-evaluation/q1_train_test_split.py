"""Q1 (Easy): 80/20 stratified split with a fixed random_state.

Prints the shape and class balance of both parts.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Load data as a DataFrame so class names are easy to read
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# stratify=y keeps the class ratio the same in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"Full data : X={X.shape}, y={y.shape}")
print(f"Train     : X={X_train.shape}, y={y_train.shape}")
print(f"Test      : X={X_test.shape}, y={y_test.shape}\n")

balance = pd.DataFrame(
    {
        "full": y.value_counts(normalize=True),
        "train": y_train.value_counts(normalize=True),
        "test": y_test.value_counts(normalize=True),
    }
).round(3)
balance.index = [data.target_names[i] for i in balance.index]
print("Class balance (proportion):")
print(balance)
