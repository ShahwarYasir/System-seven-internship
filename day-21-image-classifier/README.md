# Day 21 - Image Classifier (Transfer Learning)

**System 7 | Week 3 | Module D (Day 7)**
Topics: multiclass classification, transfer learning with MobileNetV2, feature extraction vs fine-tuning, model evaluation, prediction visualization. This folder is the **image classifier** deliverable: a 10-class CIFAR-10 classifier.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q18_transfer_learning_mobilenetv2.py` | Q18 | Hard | MobileNetV2 with a frozen base, compared with the Day 20 scratch CNN, with prediction visualizations |
| `bonus_fine_tuning.py` | Learning topic | | Unfreezes the top of MobileNetV2 and compares fine-tuning with feature extraction |
| `predict_image.py` | Practical task | | Classifies any image file you give it with the trained model |
| `fill_readme.py` | Helper | | Fills the result numbers in this README from the saved JSON files |
| `outputs/` | | | Plots, results JSON files and the saved model |

## How to run

On a GPU (Colab T4 works), because the 96x96 inputs are slow on a CPU:

```bash
pip install -r requirements.txt
# put cnn_cifar10.keras (from day-20-cnn-basics/outputs/) next to the scripts for the comparison
python q18_transfer_learning_mobilenetv2.py
python bonus_fine_tuning.py
python fill_readme.py                          # writes the numbers into README.md
python predict_image.py my_photo.jpg           # try your own picture
```

Add `QUICK_TEST=1` in front of a script for a one-minute smoke test on a tiny subset (results go to `outputs_quick/` and are not meaningful).

## Results

### Q18: MobileNetV2 (frozen base) vs CNN from scratch

| Model | Total params | Trainable params | Test accuracy |
|---|---|---|---|
| CNN from scratch (Day 20) | 591,274 | 591,274 | 0.7693 |
| MobileNetV2, frozen base | 2,270,794 | 12,810 | 0.8625 |

MobileNetV2 with a frozen base scored 0.8625, which is 9.32 percentage points higher than the CNN from scratch (0.7693).

- **Setup:** images are upscaled from 32x32 to 96x96 inside the model and scaled to the -1 to 1 range that MobileNetV2 expects. Only the new head (global average pooling, dropout 0.2, 10-way softmax) is trained, with Adam at 0.001 and early stopping. It ran for 9 epochs in about 97 seconds. Test loss 0.3954.
- **Best classes (F1):** automobile (0.934), truck (0.921), ship (0.921).
- **Worst classes (F1):** cat (0.732), dog (0.804), deer (0.828).
- **Most common mistake:** cat predicted as dog (129 test images).

Both models are evaluated on the same 10,000 official test images, so the comparison is fair. Plots: `outputs/q18_training_curves.png`, `outputs/q18_confusion_matrix.png`.

### Prediction visualizations

- `outputs/q18_predictions.png`: 12 random test images with the true label, the MobileNetV2 prediction and its confidence, and the scratch CNN's prediction side by side. The title is green when MobileNetV2 is right.
- `outputs/q18_confident_mistakes.png`: the 12 images MobileNetV2 got wrong with the highest confidence. These show where the model is confidently wrong, which is often an image that is ambiguous even to a person at 32x32 pixels.

### Fine-tuning (bonus)

Unfreezing the last 30 layers of MobileNetV2 and training them at a learning rate of 5e-05 changed test accuracy from 0.8625 to 0.8883 (+2.58 percentage points), with 1,539,210 trainable parameters instead of 12,810. Plot: `outputs/bonus_fine_tuning.png`.

## What the comparison tells us

- **The pretrained base clearly won.** With only 12,810 trainable parameters it beat the scratch CNN by about 9 points, and fine-tuning added another 2.6 points (0.8883 in total, about 12 points above the scratch CNN). The likely reason is that the base already knows how to see edges, textures and object parts from about 1.4 million ImageNet images, while the scratch CNN has to learn every filter from 45,000 small images.
- **Both models struggle with the same classes.** Cat and dog are the weakest classes for both (scratch CNN F1 0.601 and 0.677, MobileNetV2 0.732 and 0.804), and cat/dog confusion is the most common mistake in both. Transfer learning reduced these errors but did not remove them, because at 32x32 pixels the two look alike.
- **Upscaling adds no detail.** CIFAR-10 images are tiny, and resizing them to 96x96 lets the pretrained filters work at a scale they know but does not add real information. A higher input size or the original photos would probably help further.
- **It is not a perfectly controlled comparison.** MobileNetV2 is a larger network than the scratch CNN (2.27M vs 0.59M parameters in total) and uses a different input size, and each result comes from a single run. The 9-point gap is large enough to trust, but a difference of a point or less would not be.
- **Fine-tuning costs more.** It trains 1,539,210 parameters instead of 12,810 and needs a very small learning rate (0.00005) so the pretrained weights are adjusted rather than damaged.

## Learning note: transfer learning

Transfer learning means starting from a network that was already trained on a very large dataset (here ImageNet, with about 1.4 million images) instead of from random weights. The early layers of such a network learn generic visual building blocks (edges, colours, textures), and only the last layers are specific to the original task. Those generic parts are useful for almost any image problem.

| | Feature extraction | Fine-tuning |
|---|---|---|
| What changes | Base is frozen, only a new classification head is trained | The top layers of the base are unfrozen and trained too |
| Learning rate | Normal (0.001) | Very small (here 0.00005), so pretrained weights are nudged, not destroyed |
| Cost | Cheap and fast | Slower |
| Risk | Underfitting if the new task is very different | Overfitting or damaging good features if the data is small or the rate is too high |
| Best when | Little data, and the task is similar to the pretraining data | More data, or a task that differs from the pretraining data |

Practical details used here: the base is called with `training=False` so its BatchNorm layers stay in inference mode, and the model must be re-compiled after changing which layers are trainable.

Analogy: feature extraction is hiring an experienced chef and only teaching them your menu. Fine-tuning also lets them adjust their core techniques to your kitchen, which can help but risks losing what made them good if you push too hard.
