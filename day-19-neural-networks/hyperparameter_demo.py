"""Learning-topic demo: learning rate, batch size and epochs.

Trains the Q15 network several times, changing one setting at a time, and
records validation loss. Uses the breast cancer data and the Adam optimizer.
"""
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
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
sc = StandardScaler().fit(X_train)
X_train, X_test = sc.transform(X_train), sc.transform(X_test)


def train(lr, batch_size, epochs=40):
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([layers.Input(shape=(30,)),
                          layers.Dense(16, activation="relu"),
                          layers.Dense(8, activation="relu"),
                          layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    h = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs,
              batch_size=batch_size, verbose=0)
    return h.history, m.evaluate(X_test, y_test, verbose=0)[1]


fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

print("Learning rate (batch size 32):")
for lr in [0.0001, 0.001, 0.01, 0.1]:
    h, acc = train(lr, 32)
    axes[0].plot(h["val_loss"], label=f"lr={lr}")
    print(f"  lr={lr:<7} final val loss={h['val_loss'][-1]:.4f}  test acc={acc:.4f}")
axes[0].set_title("Learning rate"); axes[0].set_ylabel("Validation loss")

print("\nBatch size (lr 0.001): updates per epoch = 364 / batch size")
for bs in [8, 32, 128, 364]:
    h, acc = train(0.001, bs)
    axes[1].plot(h["val_loss"], label=f"batch={bs}")
    print(f"  batch={bs:<4} final val loss={h['val_loss'][-1]:.4f}  test acc={acc:.4f}")
axes[1].set_title("Batch size")

print("\nEpochs (lr 0.001, batch 32): train vs validation loss over 150 epochs")
h, acc = train(0.001, 32, epochs=150)
axes[2].plot(h["loss"], label="train"); axes[2].plot(h["val_loss"], label="validation")
best = int(np.argmin(h["val_loss"])) + 1
axes[2].axvline(best - 1, color="gray", ls="--", label=f"best epoch = {best}")
axes[2].set_title("Number of epochs")
print(f"  best validation epoch = {best}, final train loss={h['loss'][-1]:.4f}, "
      f"final val loss={h['val_loss'][-1]:.4f}")

for ax in axes:
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/hyperparameter_demo.png", dpi=150)
print("Saved outputs/hyperparameter_demo.png")
