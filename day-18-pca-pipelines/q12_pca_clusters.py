"""Q12 (Hard): Reduce data to 2 principal components, plot the clusters and
report the explained variance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

data = load_wine()   # 178 samples, 13 features, 3 wine classes
X = StandardScaler().fit_transform(data.data)   # PCA needs scaled features
y = data.target

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

ratio = pca.explained_variance_ratio_
print(f"Original shape: {X.shape} -> reduced shape: {X_2d.shape}")
print(f"PC1 explains {ratio[0]*100:.2f}% of the variance")
print(f"PC2 explains {ratio[1]*100:.2f}% of the variance")
print(f"Together     {ratio.sum()*100:.2f}%  (information lost: {100 - ratio.sum()*100:.2f}%)")

# Cluster in the 2D space and compare with the true classes
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_2d)
print(f"K-Means (k=3) vs true classes, ARI = {adjusted_rand_score(y, labels):.3f}")

# Top features driving each component
for i in range(2):
    top = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
    print(f"PC{i+1} top features:", [data.feature_names[j] for j in top])

full = PCA().fit(X)
cum = np.cumsum(full.explained_variance_ratio_)
print(f"Components needed for 95% variance: {int(np.argmax(cum >= 0.95)) + 1} of {X.shape[1]}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for cls, name in enumerate(data.target_names):
    axes[0].scatter(*X_2d[y == cls].T, label=name, s=25)
axes[0].set_title("PCA projection - true classes")
axes[0].legend()
axes[1].scatter(*X_2d.T, c=labels, cmap="viridis", s=25)
axes[1].set_title("PCA projection - K-Means clusters")
for ax in axes[:2]:
    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
axes[2].plot(range(1, len(cum) + 1), cum, "o-")
axes[2].axhline(0.95, color="gray", ls="--", label="95%")
axes[2].set_title("Cumulative explained variance")
axes[2].set_xlabel("Number of components")
axes[2].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q12_pca_clusters.png", dpi=150)
print("Saved outputs/q12_pca_clusters.png")
