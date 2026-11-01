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
| CNN from scratch (Day 20) | {{SC_PARAMS}} | {{SC_PARAMS}} | {{SC_ACC}} |
| MobileNetV2, frozen base | {{MN_TOTAL}} | {{MN_TRAIN}} | {{MN_ACC}} |

{{COMPARISON}}

- **Setup:** images are upscaled from 32x32 to {{IMG}}x{{IMG}} inside the model and scaled to the -1 to 1 range that MobileNetV2 expects. Only the new head (global average pooling, dropout 0.2, 10-way softmax) is trained, with Adam at 0.001 and early stopping. It ran for {{EPOCHS}} epochs in about {{SECONDS}} seconds. Test loss {{MN_LOSS}}.
- **Best classes (F1):** {{BEST3}}.
- **Worst classes (F1):** {{WORST3}}.
- **Most common mistake:** {{TOP_MISTAKE}}.

Both models are evaluated on the same 10,000 official test images, so the comparison is fair. Plots: `outputs/q18_training_curves.png`, `outputs/q18_confusion_matrix.png`.

### Prediction visualizations

- `outputs/q18_predictions.png`: 12 random test images with the true label, the MobileNetV2 prediction and its confidence, and the scratch CNN's prediction side by side. The title is green when MobileNetV2 is right.
- `outputs/q18_confident_mistakes.png`: the 12 images MobileNetV2 got wrong with the highest confidence. These show where the model is confidently wrong, which is often an image that is ambiguous even to a person at 32x32 pixels.

### Fine-tuning (bonus)

{{FINE_TUNING}}

## What the comparison does and does not tell us

- A pretrained base is not guaranteed to win. It was trained on large 224x224 photos of ImageNet, while CIFAR-10 images are tiny. Upscaling to 96x96 lets the pretrained filters work at a scale they know, but it adds no new detail. The scratch CNN learns directly from the native 32x32 pixels.
- The frozen model trains very few parameters (only the head), so it is cheap and stable. The scratch CNN has to learn every filter from 45,000 images.
- Results come from a single run each, so a gap of a point or less should not be read as a real difference.

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
