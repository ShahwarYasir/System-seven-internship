"""Practical task (Module C): Predict student results with a neural network.

Predicts pass/fail from study habits and past performance. The data is
synthetic (generated below), so a logistic-regression baseline is trained too,
to check whether the neural network adds anything over a simple model.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
rng = np.random.default_rng(SEED)

# --- 1. Synthetic student data --------------------------------------------
n = 1500
df = pd.DataFrame({
    "study_hours": rng.uniform(0, 25, n),          # per week
    "attendance": rng.uniform(40, 100, n),         # percent
    "prev_gpa": rng.uniform(1.5, 4.0, n),
    "assignment_avg": rng.uniform(30, 100, n),     # percent
    "sleep_hours": rng.uniform(4, 9, n),
    "absences": rng.poisson(4, n),
})
# Hidden rule: mostly linear, plus a non-linear study/sleep interaction and noise
score = (0.12 * df.study_hours + 0.03 * df.attendance + 0.9 * df.prev_gpa
         + 0.025 * df.assignment_avg - 0.08 * df.absences
         + 0.15 * np.minimum(df.sleep_hours, 7.5) * (df.study_hours > 8)
         + rng.normal(0, 0.6, n))
df["passed"] = (score > score.quantile(0.40)).astype(int)    # ~60% pass
print("Pass rate:", round(df["passed"].mean(), 3))

features = [c for c in df.columns if c != "passed"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["passed"], test_size=0.2, random_state=SEED, stratify=df["passed"]
)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# --- 2. Baseline -----------------------------------------------------------
base = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
base_pred = base.predict(X_test_s)

# --- 3. Neural network -----------------------------------------------------
keras.utils.set_random_seed(SEED)
model = keras.Sequential([
    layers.Input(shape=(len(features),)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                           restore_best_weights=True)
hist = model.fit(X_train_s, y_train, validation_split=0.2, epochs=200,
                 batch_size=32, callbacks=[early_stop], verbose=0)
print(f"\nTrained for {len(hist.history['loss'])} epochs "
      f"(best epoch {int(np.argmin(hist.history['val_loss'])) + 1})")

nn_pred = (model.predict(X_test_s, verbose=0).ravel() >= 0.5).astype(int)

# --- 4. Compare ------------------------------------------------------------
print("\nTest-set comparison:")
print(f"{'Model':<22}{'Accuracy':>10}{'Recall(fail)':>14}{'F1(pass)':>10}")
for name, pred in [("Logistic Regression", base_pred), ("Neural Network", nn_pred)]:
    print(f"{name:<22}{(pred == y_test).mean():>10.4f}"
          f"{recall_score(y_test, pred, pos_label=0):>14.4f}{f1_score(y_test, pred):>10.4f}")

print("\nNeural network classification report:")
print(classification_report(y_test, nn_pred, target_names=["fail", "pass"], digits=3))

# --- 5. Plots --------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(hist.history["loss"], label="train")
axes[0].plot(hist.history["val_loss"], label="validation")
axes[0].set_title("Loss curves"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(hist.history["accuracy"], label="train")
axes[1].plot(hist.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
ConfusionMatrixDisplay.from_predictions(y_test, nn_pred, display_labels=["fail", "pass"],
                                        cmap="Blues", ax=axes[2], colorbar=False)
axes[2].set_title("Neural network - confusion matrix")
for ax in axes[:2]:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/student_results_nn.png", dpi=150)
print("Saved outputs/student_results_nn.png")
