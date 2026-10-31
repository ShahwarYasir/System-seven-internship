# Day 20 - CNN Basics

**System 7 | Week 3 | Module D (Day 6)**
Topics: image preprocessing, data augmentation, convolution and pooling layers, CNN on CIFAR-10, why CNNs suit images.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q16_preprocessing_augmentation.py` | Q16 | Medium | Resize to 64x64, normalize, apply augmentation layers, show 9 augmented samples |
| `q17_cnn_cifar10.py` | Q17 | Hard | CNN with Conv2D and MaxPooling on CIFAR-10, test accuracy and confusion matrix |
| `bonus_dense_vs_cnn.py` | Learning topic | | Fully connected network vs the Q17 CNN (run Q17 first) |
| `outputs/` | | | Plots, `q17_results.json`, and the saved model `cnn_cifar10.keras` (used again on Day 21) |

## How to run

```bash
pip install -r requirements.txt
mkdir -p outputs
python q16_preprocessing_augmentation.py
python q17_cnn_cifar10.py              # slow on a CPU: about 80 s per epoch
python bonus_dense_vs_cnn.py           # needs outputs/cnn_cifar10.keras from Q17
```

CIFAR-10 downloads automatically through Keras the first time.

## Results

**Q16.** A CIFAR-10 test image (the first cat) goes through two stages:
1. Preprocessing: `Resizing(64, 64)` and `Rescaling(1/255)` turn a 32x32 uint8 image with values 0 to 255 into a 64x64 float image with values 0 to 1.
2. Augmentation: random horizontal flip, rotation (up to 10% of a turn), zoom (15%), translation (10%) and contrast (20%).

These layers only act when called with `training=True`. At prediction time they do nothing, so the model sees clean images. Because CIFAR images are only 32x32, upscaling to 64x64 makes them smoother but adds no real detail. Plot: `outputs/q16_augmentation.png`.

**Q17.** Architecture: two blocks of (Conv 3x3, Conv 3x3, MaxPool, Dropout) with 32 then 64 filters, then Flatten, Dense 128, Dropout 0.5 and a 10-way softmax. It has 591,274 parameters, uses Adam, and stops early on validation loss.

- **Test accuracy: 0.7693** (test loss 0.6743) on the 10,000 official CIFAR-10 test images, trained on Colab. Early stopping ended training after 16 epochs.
- Best classes: ship (F1 0.882), automobile (0.873), truck (0.849).
- Worst classes: cat (F1 0.601), bird (0.640), dog (0.677).
- Most common mistake: dog predicted as cat (209 test images). Animals that share pose, fur and background are hard to separate at 32x32 pixels, while vehicles have distinctive shapes.

Plots: `outputs/q17_training_curves.png`, `outputs/q17_confusion_matrix.png`.

Other frequent confusions are cat predicted as dog and automobile predicted as truck (138 images each). Training and validation accuracy finish together at about 0.775, and validation sits above training early on because dropout only acts during training, so there is no sign of overfitting. The model is limited by its size and the 32x32 resolution rather than by memorising, which better augmentation and transfer learning (Day 21) can improve.


## Learning note: why CNNs suit images

An image is a grid where nearby pixels belong together. A fully connected network ignores this: it flattens the picture into a list and links every pixel to every unit. A CNN uses three ideas instead:

- **Local connections:** each filter looks at a small patch (3x3), which is where edges and textures live.
- **Weight sharing:** the same small filter slides across the whole image, so a cat's ear is detected wherever it appears, with few parameters.
- **Pooling:** shrinks the feature maps (32x32 to 16x16 to 8x8), keeping the strongest responses. This cuts computation and tolerates small shifts.

Early layers learn edges, middle layers learn textures and parts, and deeper layers learn whole objects.

**Evidence from `bonus_dense_vs_cnn.py`** (same data, 10 epochs for the dense network):

| Model | Parameters | Test accuracy |
|---|---|---|
| Dense network (512, 256, 10) | 1,707,274 | 0.4694 |
| CNN from Q17 | 591,274 | 0.7693 |

The CNN has about a third of the parameters and scores roughly 30 points higher.

> The dense-network row came from an earlier run on a copy of CIFAR-10. After running `bonus_dense_vs_cnn.py` on your own data, replace that accuracy with your printed value. The first layer shows why: a dense layer of 512 units on a flattened 32x32x3 image needs 1,573,376 parameters, while a Conv2D layer with 32 filters of 3x3 needs only 896.

Analogy: a dense network is a detective who memorises each photo pixel by pixel. A CNN is a detective with a magnifying glass who learns what an ear or a wheel looks like and then scans every photo for it.
