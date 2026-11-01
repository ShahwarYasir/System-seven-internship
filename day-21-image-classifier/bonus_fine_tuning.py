"""Bonus (learning topic): Feature extraction vs fine-tuning.

Loads the frozen-base model from Q18, unfreezes the last 30 layers of MobileNetV2
and trains them with a very small learning rate. Compares test accuracy before
and after. Run q18_transfer_learning_mobilenetv2.py first.

Quick smoke test:  QUICK_TEST=1 python bonus_fine_tuning.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

SEED = 42
BATCH = 128
EPOCHS = 6
UNFREEZE_LAST = 30
LEARNING_RATE = 5e-5          # tiny: pretrained weights should only be nudged
QUICK = os.environ.get("QUICK_TEST") == "1"
OUT = "outputs_quick" if QUICK else "outputs"
keras.utils.set_random_seed(SEED)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.ravel(), y_test.ravel()
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
)
if QUICK:
    x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=2000, stratify=y_train, random_state=SEED)
    x_val, _, y_val, _ = train_test_split(x_val, y_val, train_size=500, stratify=y_val, random_state=SEED)
    x_test, _, y_test, _ = train_test_split(x_test, y_test, train_size=1000, stratify=y_test, random_state=SEED)
    EPOCHS = 1


def make_ds(x, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED)
    return ds.batch(BATCH).map(lambda a, b: (tf.cast(a, tf.float32), b)).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = make_ds(x_train, y_train, True), make_ds(x_val, y_val), make_ds(x_test, y_test)

model = keras.models.load_model(f"{OUT}/mobilenetv2_frozen.keras")
before_loss, before_acc = model.evaluate(test_ds, verbose=0)
print(f"Before fine-tuning (frozen base): test accuracy = {before_acc:.4f}")

# Unfreeze only the top of the pretrained base; early layers hold generic edges and textures
base = next(l for l in model.layers if isinstance(l, keras.Model))
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False
trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
print(f"Unfroze the last {UNFREEZE_LAST} layers of the base -> trainable parameters: {trainable_params:,}")

model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),   # re-compile after changing trainable
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop], verbose=2)

after_loss, after_acc = model.evaluate(test_ds, verbose=0)
print(f"\nAfter fine-tuning: test accuracy = {after_acc:.4f}  "
      f"(change {100 * (after_acc - before_acc):+.2f} percentage points)")

print("\n--- Ready-to-paste README table ---")
print("| Stage | Trainable params | Test accuracy |")
print("|---|---|---|")
print(f"| Feature extraction (frozen base) | see Q18 | {before_acc:.4f} |")
print(f"| Fine-tuning (last {UNFREEZE_LAST} layers, lr={LEARNING_RATE}) | {trainable_params:,} | {after_acc:.4f} |")

plt.figure(figsize=(6, 4))
plt.plot(hist.history["accuracy"], label="train"); plt.plot(hist.history["val_accuracy"], label="validation")
plt.axhline(before_acc, color="gray", ls="--", label=f"frozen test acc {before_acc:.3f}")
plt.title("Fine-tuning accuracy"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/bonus_fine_tuning.png", dpi=150)
with open(f"{OUT}/bonus_fine_tuning_results.json", "w") as f:
    json.dump({"before_test_accuracy": float(before_acc), "after_test_accuracy": float(after_acc),
               "trainable_params": trainable_params, "unfrozen_layers": UNFREEZE_LAST,
               "learning_rate": LEARNING_RATE, "epochs_run": len(hist.history["loss"]),
               "quick_test": QUICK}, f, indent=2)
print(f"Saved plot and results to {OUT}/")
