import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(x, np.sin(x), label="sin", color="tab:blue")
axes[0, 0].plot(x, np.cos(x), label="cos", color="tab:orange", linestyle="--")
axes[0, 0].set_title("Line chart")
axes[0, 0].set_xlabel("x")
axes[0, 0].legend()

axes[0, 1].bar(["CS", "AI", "SE", "EE"], [58, 61, 50, 33], color="tab:green")
axes[0, 1].set_title("Bar chart")
axes[0, 1].set_ylabel("Students")

rng = np.random.default_rng(1)
axes[1, 0].scatter(rng.normal(size=60), rng.normal(size=60), alpha=0.7)
axes[1, 0].set_title("Scatter plot")

axes[1, 1].hist(rng.normal(50, 10, 500), bins=20, color="tab:purple", edgecolor="white")
axes[1, 1].set_title("Histogram")

plt.tight_layout()
plt.savefig("matplotlib_basics.png", dpi=120)
plt.show()
