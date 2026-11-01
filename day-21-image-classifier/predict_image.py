"""Classify your own image with the trained MobileNetV2 model.

Usage:  python predict_image.py path/to/photo.jpg [path/to/model.keras]
The model was trained on 32x32 CIFAR-10 images, so it only knows these classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
"""
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

if len(sys.argv) < 2:
    sys.exit(__doc__)
image_path = sys.argv[1]
model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/mobilenetv2_frozen.keras"

model = keras.models.load_model(model_path)
img = Image.open(image_path).convert("RGB").resize((32, 32))   # same size the model was trained on
probs = model.predict(np.array(img, dtype="float32")[None, ...], verbose=0)[0]

print(f"Top predictions for {image_path}:")
for k in np.argsort(probs)[::-1][:3]:
    print(f"  {CLASS_NAMES[k]:<11} {probs[k] * 100:5.1f}%")
