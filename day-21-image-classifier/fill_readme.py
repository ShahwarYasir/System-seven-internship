"""Fill the results placeholders in README.md from the JSON files the scripts saved.

Run this after q18_transfer_learning_mobilenetv2.py (and optionally
bonus_fine_tuning.py) have finished:   python fill_readme.py
It reads README.template.md and writes README.md.
"""
import json
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "outputs"
q18 = json.load(open(f"{out}/q18_results.json"))
ft_path = f"{out}/bonus_fine_tuning_results.json"
ft = json.load(open(ft_path)) if os.path.exists(ft_path) else None

f1 = sorted(q18["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True)
fmt = lambda items: ", ".join(f"{name} ({v:.3f})" for name, v in items)

mn, sc = q18["mobilenetv2_test_accuracy"], q18["scratch_test_accuracy"]
if sc is None:
    comparison = ("The scratch CNN was not found when Q18 ran, so no comparison was made. "
                  "Copy `cnn_cifar10.keras` next to the script and run it again.")
    sc_acc, sc_params = "n/a", "n/a"
else:
    diff = 100 * (mn - sc)
    word = "higher" if diff > 0 else "lower"
    comparison = (f"MobileNetV2 with a frozen base scored {mn:.4f}, which is {abs(diff):.2f} percentage points "
                  f"{word} than the CNN from scratch ({sc:.4f}).")
    sc_acc, sc_params = f"{sc:.4f}", f"{q18['scratch_params']:,}"

if ft:
    d = 100 * (ft["after_test_accuracy"] - ft["before_test_accuracy"])
    ft_text = (f"Unfreezing the last {ft['unfrozen_layers']} layers of MobileNetV2 and training them at a learning rate of "
               f"{ft['learning_rate']} changed test accuracy from {ft['before_test_accuracy']:.4f} to "
               f"{ft['after_test_accuracy']:.4f} ({d:+.2f} percentage points), with {ft['trainable_params']:,} trainable "
               f"parameters instead of {q18['mobilenetv2_trainable_params']:,}. Plot: `outputs/bonus_fine_tuning.png`.")
else:
    ft_text = "`bonus_fine_tuning.py` has not been run yet."

values = {
    "MN_ACC": f"{mn:.4f}", "MN_LOSS": f"{q18['mobilenetv2_test_loss']:.4f}",
    "MN_TOTAL": f"{q18['mobilenetv2_total_params']:,}", "MN_TRAIN": f"{q18['mobilenetv2_trainable_params']:,}",
    "EPOCHS": str(q18["epochs_run"]), "SECONDS": f"{q18['train_seconds']:.0f}", "IMG": str(q18["img_size"]),
    "SC_ACC": sc_acc, "SC_PARAMS": sc_params, "COMPARISON": comparison,
    "BEST3": fmt(f1[:3]), "WORST3": fmt(f1[-3:][::-1]),
    "TOP_MISTAKE": f"{q18['top_mistake'][0]} predicted as {q18['top_mistake'][1]} ({q18['top_mistake'][2]} test images)",
    "FINE_TUNING": ft_text,
}
text = open("README.template.md").read()
for k, v in values.items():
    text = text.replace("{{" + k + "}}", v)
open("README.md", "w").write(text)
left = [w for w in text.split("{{")[1:]]
print("README.md written." + (f" Unfilled placeholders: {len(left)}" if left else " All placeholders filled."))
