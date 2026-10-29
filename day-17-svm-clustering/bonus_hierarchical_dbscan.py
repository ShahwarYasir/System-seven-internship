"""Bonus (Module B topic): Hierarchical clustering and DBSCAN.

Compares K-Means, agglomerative clustering and DBSCAN on two shapes:
round blobs (K-Means should do well) and two moons (DBSCAN should win).
Also draws a dendrogram. Adjusted Rand Index (ARI) measures agreement with the
true labels (1.0 = perfect).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

datasets = {
    "Blobs": make_blobs(n_samples=400, centers=3, cluster_std=0.9, random_state=42),
    "Moons": make_moons(n_samples=400, noise=0.07, random_state=42),
}
eps = {"Blobs": 0.5, "Moons": 0.3}

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for row, (name, (X, y)) in enumerate(datasets.items()):
    X = StandardScaler().fit_transform(X)
    k = len(set(y))
    results = {
        "K-Means": KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X),
        "Hierarchical (ward)": AgglomerativeClustering(n_clusters=k).fit_predict(X),
        "DBSCAN": DBSCAN(eps=eps[name], min_samples=5).fit_predict(X),
    }
    for col, (algo, labels) in enumerate(results.items()):
        ari = adjusted_rand_score(y, labels)
        n_noise = int((labels == -1).sum())
        print(f"{name:<6} {algo:<20} ARI={ari:.3f}  clusters={len(set(labels) - {-1})}  noise={n_noise}")
        axes[row, col].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=12)
        axes[row, col].set_title(f"{name}: {algo}\nARI = {ari:.2f}")
plt.tight_layout()
plt.savefig("outputs/bonus_cluster_comparison.png", dpi=150)

# Dendrogram on a small sample so it stays readable
Xb, _ = make_blobs(n_samples=40, centers=3, cluster_std=0.9, random_state=42)
plt.figure(figsize=(9, 4))
dendrogram(linkage(StandardScaler().fit_transform(Xb), method="ward"))
plt.title("Dendrogram (ward linkage, 40 samples)")
plt.xlabel("Sample index")
plt.ylabel("Merge distance")
plt.tight_layout()
plt.savefig("outputs/bonus_dendrogram.png", dpi=150)
print("Saved outputs/bonus_cluster_comparison.png and outputs/bonus_dendrogram.png")
