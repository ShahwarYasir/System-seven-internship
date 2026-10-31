"""Q17 (Hard): Build a CNN with Conv2D and MaxPooling layers for CIFAR-10;
report test accuracy and a confusion matrix.

The trained model is saved to outputs/cnn_cifar10.keras so that Day 21 can
compare it against a transfer-learning model.
"""
import os
os.makedirs("outputs", exist_ok=True)  # create the folder if it is missing
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 20
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
keras.utils.set_random_seed(SEED)

# --- Data: 50,000 train / 10,000 test images, 32x32 RGB --------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0          # normalize to 0..1
x_test = x_test.astype("float32") / 255.0
y_train, y_test = y_train.ravel(), y_test.ravel()
# Hold out 10% of the training images for validation. A stratified split keeps all
# 10 classes in the validation set whatever the order of the data.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
)
print("Train:", x_train.shape, "Validation:", x_val.shape, "Test:", x_test.shape)

# --- Model: Conv -> Conv -> Pool blocks, then a small dense classifier ------
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),                            # 32x32 -> 16x16
    layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),                            # 16x16 -> 8x8
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),           # 10 class probabilities
])
model.summary()
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                           restore_best_weights=True)
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS,
                 batch_size=64, callbacks=[early_stop], verbose=2)

# --- Evaluate --------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTEST ACCURACY = {test_acc:.4f}   TEST LOSS = {test_loss:.4f}")

y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

cm = confusion_matrix(y_test, y_pred)
off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most common mistake: true '{CLASS_NAMES[i]}' predicted as '{CLASS_NAMES[j]}' ({off[i, j]} images)")

# --- Plots -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist.history["loss"], label="train"); axes[0].plot(hist.history["val_loss"], label="validation")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
axes[1].plot(hist.history["accuracy"], label="train"); axes[1].plot(hist.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q17_training_curves.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title(f"CIFAR-10 confusion matrix (test accuracy {test_acc:.3f})")
plt.tight_layout()
plt.savefig("outputs/q17_confusion_matrix.png", dpi=150)

model.save("outputs/cnn_cifar10.keras")
with open("outputs/q17_results.json", "w") as f:
    json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss),
               "epochs_run": len(hist.history["loss"]),
               "params": int(model.count_params())}, f, indent=2)
print("Saved outputs/q17_training_curves.png, q17_confusion_matrix.png, cnn_cifar10.keras, q17_results.json")
