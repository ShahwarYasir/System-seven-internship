"""Q16 (Medium): Resize images to 64x64, normalize them, apply augmentation
layers and display 9 augmented samples.

Uses CIFAR-10 test images (32x32 originals), so resizing to 64x64 is an
upscale. With your own photos the same layers work on any size.
"""
import os
os.makedirs("outputs", exist_ok=True)  # create the folder if it is missing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
TARGET_CLASS = "cat"    # augment the first test image of this class; change it to try others
keras.utils.set_random_seed(42)

(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
IMG_INDEX = int(np.where(y_test.ravel() == CLASS_NAMES.index(TARGET_CLASS))[0][0])
image = x_test[IMG_INDEX]                        # uint8, shape (32, 32, 3), values 0..255
print(f"Original image : shape={image.shape}, dtype={image.dtype}, "
      f"min={image.min()}, max={image.max()}, class={CLASS_NAMES[int(y_test[IMG_INDEX][0])]}")

# --- Step 1: resize to 64x64 and normalize to 0..1 --------------------------
preprocess = keras.Sequential([
    layers.Resizing(64, 64),
    layers.Rescaling(1.0 / 255),
], name="preprocess")
batch = preprocess(tf.convert_to_tensor(image[None, ...], dtype=tf.float32))
print(f"After preprocess: shape={tuple(batch.shape)}, dtype={batch.dtype.name}, "
      f"min={float(tf.reduce_min(batch)):.3f}, max={float(tf.reduce_max(batch)):.3f}")

# --- Step 2: random augmentation layers (only active when training=True) ----
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),                 # up to +/-10% of a full turn (36 degrees)
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.20),
], name="augment")

# --- Show the original, the preprocessed image, and 9 augmented samples ----
fig, axes = plt.subplots(3, 4, figsize=(11, 8))
for ax in axes.ravel():
    ax.axis("off")

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original 32x32")
axes[1, 0].imshow(batch[0].numpy())
axes[1, 0].set_title("Resized 64x64\nnormalized 0-1")

slots = [(r, c) for r in range(3) for c in range(1, 4)]
for i, (r, c) in enumerate(slots):
    aug = augment(batch, training=True)[0].numpy()
    axes[r, c].imshow(np.clip(aug, 0, 1))
    axes[r, c].set_title(f"Augmented #{i + 1}")
plt.suptitle("Q16: preprocessing and 9 augmented samples", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/q16_augmentation.png", dpi=150)
print("Saved outputs/q16_augmentation.png")
