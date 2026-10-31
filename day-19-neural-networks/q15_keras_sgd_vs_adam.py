"""Q15 (Medium): Build, compile and train a Keras Sequential model on tabular
data, plot the loss curves and compare SGD with Adam.

Both optimizers train the same architecture from the same starting weights.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 40

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler().fit(X_train)             # fit on training data only
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)


def build_model():
    keras.utils.set_random_seed(SEED)               # same initial weights every time
    return keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),      # probability of class 1
    ])


optimizers = {
    "SGD (lr=0.01)": keras.optimizers.SGD(learning_rate=0.01),
    "Adam (lr=0.001)": keras.optimizers.Adam(learning_rate=0.001),
}

histories, summary = {}, {}
for name, opt in optimizers.items():
    model = build_model()
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    h = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS,
                  batch_size=32, verbose=0)
    histories[name] = h.history
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    # first epoch where validation loss drops below 0.15, a speed measure
    below = np.where(np.array(h.history["val_loss"]) < 0.15)[0]
    summary[name] = {
        "epochs_to_val_loss<0.15": int(below[0]) + 1 if len(below) else None,
        "final_train_loss": round(h.history["loss"][-1], 4),
        "final_val_loss": round(h.history["val_loss"][-1], 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
    }

print(f"{'Optimizer':<18}{'ep<0.15':>8}{'train loss':>12}{'val loss':>10}{'test loss':>11}{'test acc':>10}")
for name, s in summary.items():
    print(f"{name:<18}{str(s['epochs_to_val_loss<0.15']):>8}{s['final_train_loss']:>12}"
          f"{s['final_val_loss']:>10}{s['test_loss']:>11}{s['test_accuracy']:>10}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, h in histories.items():
    axes[0].plot(h["loss"], label=f"{name} train")
    axes[0].plot(h["val_loss"], "--", label=f"{name} val")
    axes[1].plot(h["val_accuracy"], label=name)
axes[0].set_title("Loss curves (solid = train, dashed = validation)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Binary cross-entropy")
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q15_sgd_vs_adam.png", dpi=150)
print("Saved outputs/q15_sgd_vs_adam.png")
