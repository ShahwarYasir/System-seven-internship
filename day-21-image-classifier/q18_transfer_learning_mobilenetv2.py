"""Q18 (Hard): MobileNetV2 transfer learning with a frozen base, compared with
the CNN trained from scratch on Day 20, plus prediction visualizations.

This is also the image classifier deliverable: a 10-class CIFAR-10 classifier.

Feature extraction = keep the pretrained MobileNetV2 frozen and train only a new
classification head on top of it.

Quick smoke test (tiny subset, 1 epoch):  QUICK_TEST=1 python q18_transfer_learning_mobilenetv2.py
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = 96            # MobileNetV2 needs at least 32; 96 is a good speed/accuracy balance
EPOCHS = 10
BATCH = 128
WEIGHTS = "imagenet"     # pretrained weights (downloaded automatically the first time)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
QUICK = os.environ.get("QUICK_TEST") == "1"
OUT = "outputs_quick" if QUICK else "outputs"
os.makedirs(OUT, exist_ok=True)
keras.utils.set_random_seed(SEED)

# --- Data ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.ravel(), y_test.ravel()
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
)
if QUICK:   # tiny stratified subsets so the whole script runs in about a minute
    x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=2000, stratify=y_train, random_state=SEED)
    x_val, _, y_val, _ = train_test_split(x_val, y_val, train_size=500, stratify=y_val, random_state=SEED)
    x_test, _, y_test, _ = train_test_split(x_test, y_test, train_size=1000, stratify=y_test, random_state=SEED)
    EPOCHS = 1
print(f"Train {x_train.shape}  Validation {x_val.shape}  Test {x_test.shape}  (quick={QUICK})")


def make_ds(x, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED)
    # keep images as uint8 in memory, cast per batch
    return ds.batch(BATCH).map(lambda a, b: (tf.cast(a, tf.float32), b)).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = make_ds(x_train, y_train, True), make_ds(x_val, y_val), make_ds(x_test, y_test)

# --- Model: frozen MobileNetV2 base + new classification head ---------------
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=WEIGHTS
)
base.trainable = False                                 # freeze: its weights will not change

inputs = keras.Input(shape=(32, 32, 3))                # raw CIFAR pixels, 0..255
x = layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)        # upscale 32x32 -> 96x96
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)        # MobileNetV2 expects -1..1
x = base(x, training=False)                            # keep BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs, name="mobilenetv2_frozen")
model.summary()

trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
print(f"\nTotal parameters: {model.count_params():,}   Trainable: {trainable_params:,}")

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

t0 = time.perf_counter()
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop], verbose=2)
train_seconds = time.perf_counter() - t0
epochs_run = len(hist.history["loss"])

# --- Evaluate the transfer model -------------------------------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
probs = model.predict(test_ds, verbose=0)
pred = probs.argmax(axis=1)
print(f"\nMobileNetV2 (frozen)  TEST ACCURACY = {test_acc:.4f}   TEST LOSS = {test_loss:.4f}")
print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
per_class_f1 = classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

cm = confusion_matrix(y_test, pred)
off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most common mistake: true '{CLASS_NAMES[i]}' predicted as '{CLASS_NAMES[j]}' ({off[i, j]} images)")

# --- Compare with the CNN from scratch (Day 20) ----------------------------
scratch_paths = ["cnn_cifar10.keras", "outputs/cnn_cifar10.keras",
                 "../day-20-cnn-basics/outputs/cnn_cifar10.keras"]
scratch_path = next((p for p in scratch_paths if os.path.exists(p)), None)
scratch, scratch_pred, scratch_acc, scratch_params = None, None, None, None
if scratch_path:
    scratch = keras.models.load_model(scratch_path)
    s_probs = scratch.predict(x_test.astype("float32") / 255.0, batch_size=256, verbose=0)
    scratch_pred = s_probs.argmax(axis=1)
    scratch_acc = float((scratch_pred == y_test).mean())
    scratch_params = int(scratch.count_params())
    print(f"CNN from scratch ({scratch_path}) TEST ACCURACY = {scratch_acc:.4f}")
else:
    print("\n[!] cnn_cifar10.keras not found. Copy it next to this script (it is in "
          "day-20-cnn-basics/outputs/) to get the comparison.")

print("\n--- Ready-to-paste README table ---")
print("| Model | Total params | Trainable params | Test accuracy |")
print("|---|---|---|---|")
if scratch_path:
    print(f"| CNN from scratch (Day 20) | {scratch_params:,} | {scratch_params:,} | {scratch_acc:.4f} |")
print(f"| MobileNetV2, frozen base | {model.count_params():,} | {trainable_params:,} | {test_acc:.4f} |")

# --- Plots -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist.history["loss"], label="train"); axes[0].plot(hist.history["val_loss"], label="validation")
axes[0].set_title("Loss (MobileNetV2 head)")
axes[1].plot(hist.history["accuracy"], label="train"); axes[1].plot(hist.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy (MobileNetV2 head)")
for ax in axes:
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/q18_training_curves.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title(f"MobileNetV2 (frozen) confusion matrix, accuracy {test_acc:.3f}")
plt.tight_layout(); plt.savefig(f"{OUT}/q18_confusion_matrix.png", dpi=150); plt.close()

# Random test images with both models' predictions side by side
rng = np.random.default_rng(SEED)
idx = rng.choice(len(x_test), 12, replace=False)
fig, axes = plt.subplots(3, 4, figsize=(11, 10.5))
for ax, k in zip(axes.ravel(), idx):
    ax.imshow(x_test[k]); ax.axis("off")
    tl = f"MobileNetV2: {CLASS_NAMES[pred[k]]} ({probs[k].max():.2f})"
    lines = [f"True: {CLASS_NAMES[y_test[k]]}", tl]
    ok = pred[k] == y_test[k]
    if scratch is not None:
        lines.append(f"Scratch CNN: {CLASS_NAMES[scratch_pred[k]]}")
    ax.set_title("\n".join(lines), fontsize=8, color="green" if ok else "red")
plt.suptitle("Predictions on random test images (title is green when MobileNetV2 is right)")
plt.tight_layout(); plt.subplots_adjust(hspace=0.5, top=0.92); plt.savefig(f"{OUT}/q18_predictions.png", dpi=150); plt.close()

# The model's most confident mistakes
wrong = np.where(pred != y_test)[0]
worst = wrong[np.argsort(-probs[wrong].max(axis=1))[:12]]
fig, axes = plt.subplots(3, 4, figsize=(11, 10.5))
for ax, k in zip(axes.ravel(), worst):
    ax.imshow(x_test[k]); ax.axis("off")
    ax.set_title(f"True: {CLASS_NAMES[y_test[k]]}\nPred: {CLASS_NAMES[pred[k]]} ({probs[k].max():.2f})",
                 fontsize=9, color="red")
plt.suptitle("MobileNetV2: most confident mistakes")
plt.tight_layout(); plt.subplots_adjust(hspace=0.5, top=0.92); plt.savefig(f"{OUT}/q18_confident_mistakes.png", dpi=150); plt.close()

# --- Save ------------------------------------------------------------------
model.save(f"{OUT}/mobilenetv2_frozen.keras")
with open(f"{OUT}/q18_results.json", "w") as f:
    json.dump({
        "mobilenetv2_test_accuracy": float(test_acc), "mobilenetv2_test_loss": float(test_loss),
        "mobilenetv2_total_params": int(model.count_params()), "mobilenetv2_trainable_params": trainable_params,
        "epochs_run": epochs_run, "train_seconds": round(train_seconds, 1), "img_size": IMG_SIZE,
        "scratch_test_accuracy": scratch_acc, "scratch_params": scratch_params,
        "per_class_f1": {c: round(per_class_f1[c]["f1-score"], 4) for c in CLASS_NAMES},
        "top_mistake": [CLASS_NAMES[i], CLASS_NAMES[j], int(off[i, j])],
        "quick_test": QUICK,
    }, f, indent=2)
print(f"\nTraining took {train_seconds:.0f} s ({epochs_run} epochs). Saved plots, model and results to {OUT}/")
