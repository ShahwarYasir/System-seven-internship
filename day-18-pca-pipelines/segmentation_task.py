"""Practical task (Module B): Customer segmentation.

Steps: build a customer dataset -> engineer features -> scale -> PCA -> K-Means
(choose k with silhouette) -> profile and name each segment.

The data is synthetic, generated from four hidden customer types, so there is
a ground truth to sanity-check the result against.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

# --- 1. Synthetic customers from 4 hidden types ---------------------------
types = {
    #                 n,  age, income(k), visits/yr, avg basket
    "budget_regular": (180, 34, 35, 40, 18),
    "big_spender":    (120, 41, 110, 25, 95),
    "young_trendy":   (150, 24, 45, 55, 40),
    "occasional":     (150, 52, 70, 6, 60),
}
frames = []
for name, (n, age, inc, visits, basket) in types.items():
    frames.append(pd.DataFrame({
        "age": rng.normal(age, 4, n).clip(18, 75),
        "annual_income_k": rng.normal(inc, inc * 0.12, n).clip(10),
        "visits_per_year": rng.normal(visits, visits * 0.15, n).clip(1),
        "avg_basket": rng.normal(basket, basket * 0.15, n).clip(5),
        "hidden_type": name,
    }))
df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# --- 2. Feature engineering -----------------------------------------------
df["annual_spend"] = df["visits_per_year"] * df["avg_basket"]
df["spend_pct_of_income"] = 100 * df["annual_spend"] / (df["annual_income_k"] * 1000)
features = ["age", "annual_income_k", "visits_per_year", "avg_basket",
            "annual_spend", "spend_pct_of_income"]
print("Customers:", len(df), "| features:", features)

# --- 3. Scale and reduce --------------------------------------------------
X = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=0.90, random_state=42)     # keep 90% of the variance
X_pca = pca.fit_transform(X)
print(f"PCA: {X.shape[1]} features -> {X_pca.shape[1]} components "
      f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance kept)")

# --- 4. Choose k and cluster ----------------------------------------------
scores = {}
for k in range(2, 9):
    scores[k] = silhouette_score(X_pca, KMeans(k, n_init=10, random_state=42).fit_predict(X_pca))
best_k = max(scores, key=scores.get)
print("Silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
print(f"Chosen k = {best_k}")

df["segment"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(X_pca)

# --- 5. Profile the segments ----------------------------------------------
profile = df.groupby("segment")[features].mean().round(1)
profile.insert(0, "customers", df["segment"].value_counts().sort_index())
pd.set_option("display.width", 200)
print("\nSegment profiles (averages):")
print(profile.to_string())

print("\nHow segments line up with the hidden types (rows = segment):")
print(pd.crosstab(df["segment"], df["hidden_type"]).to_string())

profile.to_csv("outputs/segment_profiles.csv")

# --- 6. Plots --------------------------------------------------------------
X2 = PCA(n_components=2).fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sc = axes[0].scatter(*X2.T, c=df["segment"], cmap="viridis", s=18)
axes[0].set_title("Customer segments (PCA view)")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].legend(*sc.legend_elements(), title="Segment")
axes[1].scatter(df["visits_per_year"], df["avg_basket"], c=df["segment"], cmap="viridis", s=18)
axes[1].set_title("Visits per year vs average basket")
axes[1].set_xlabel("Visits per year"); axes[1].set_ylabel("Average basket")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/segmentation.png", dpi=150)
print("\nSaved outputs/segmentation.png and outputs/segment_profiles.csv")
