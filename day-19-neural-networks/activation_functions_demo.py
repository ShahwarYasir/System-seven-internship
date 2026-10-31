"""Learning-topic demo: ReLU, sigmoid, tanh and softmax."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-6, 6, 400)
relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)

fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, (name, y, note) in zip(axes[:3], [
    ("ReLU", relu, "hidden layers"),
    ("Sigmoid", sigmoid, "binary output"),
    ("Tanh", tanh, "zero-centred, older nets"),
]):
    ax.plot(z, y, lw=2)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"{name}\n({note})"); ax.grid(alpha=0.3)

# Softmax turns a vector of raw scores (logits) into probabilities that sum to 1
logits = np.array([2.0, 1.0, 0.1, -1.0])
softmax = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
axes[3].bar([f"class {i}" for i in range(4)], softmax, color="tab:green")
axes[3].set_title(f"Softmax (multiclass output)\nsum = {softmax.sum():.2f}")
for i, v in enumerate(softmax):
    axes[3].text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig("outputs/activation_functions.png", dpi=150)

print("logits  :", logits)
print("softmax :", np.round(softmax, 4), "-> sum =", round(softmax.sum(), 4))
print("sigmoid(0) =", round(1 / (1 + np.exp(0.0)), 3), "| relu(-3) = 0.0 | relu(3) = 3.0")
print("Saved outputs/activation_functions.png")
