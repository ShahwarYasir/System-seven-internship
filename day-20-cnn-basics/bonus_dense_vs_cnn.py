"""Bonus (learning topic): Why CNNs suit images.

Compares a plain fully connected network with the CNN from Q17 on CIFAR-10:
parameter count first, then accuracy. Run q17_cnn_cifar10.py first so that
outputs/cnn_cifar10.keras exists.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 10
keras.utils.set_random_seed(SEED)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
y_train, y_test = y_train.ravel(), y_test.ravel()
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
)

# A dense network must flatten the image, which throws away the 2D layout
dense = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),                                  # 3,072 inputs
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
dense.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
dense.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS,
          batch_size=64, verbose=2)
dense_acc = dense.evaluate(x_test, y_test, verbose=0)[1]

print("\n" + "=" * 60)
print(f"Dense network : {dense.count_params():>9,} parameters, test accuracy {dense_acc:.4f}")

if os.path.exists("outputs/cnn_cifar10.keras"):
    cnn = keras.models.load_model("outputs/cnn_cifar10.keras")
    cnn_acc = cnn.evaluate(x_test, y_test, verbose=0)[1]
    print(f"CNN (Q17)     : {cnn.count_params():>9,} parameters, test accuracy {cnn_acc:.4f}")
else:
    print("Run q17_cnn_cifar10.py first to compare with the CNN.")

# Where the parameters go: one dense layer vs one conv layer on the same input
print("\nFirst-layer parameter count on a 32x32x3 image:")
print(f"  Dense(512) on flattened input : {3072 * 512 + 512:>9,}  (every pixel linked to every unit)")
print(f"  Conv2D(32, 3x3)               : {3 * 3 * 3 * 32 + 32:>9,}  (one small filter reused across the image)")
