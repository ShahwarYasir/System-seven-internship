"""Q9 (Medium): Fit K-Means for k = 2..10, plot the elbow curve and the
silhouette scores, and use them to pick k."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Synthetic data with 4 true groups, so we can check whether the metrics find 4
X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.1, random_state=42)
X = StandardScaler().fit_transform(X)   # K-Means uses distances, so scale first

ks = list(range(2, 11))
inertia, silhouette = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, km.labels_))

print(f"{'k':>3} {'inertia':>10} {'silhouette':>11}")
for k, i, s in zip(ks, inertia, silhouette):
    print(f"{k:>3} {i:>10.2f} {s:>11.4f}")

best_k = ks[silhouette.index(max(silhouette))]
print(f"\nBest k by silhouette score: {best_k} (score {max(silhouette):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ks, inertia, "o-")
axes[0].set_title("Elbow curve")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].grid(alpha=0.3)

axes[1].plot(ks, silhouette, "s-", color="tab:orange")
axes[1].axvline(best_k, color="gray", ls="--", label=f"Best k = {best_k}")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Mean silhouette")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/q9_elbow_silhouette.png", dpi=150)
print("Saved outputs/q9_elbow_silhouette.png")
