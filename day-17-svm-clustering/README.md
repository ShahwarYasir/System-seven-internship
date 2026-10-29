# Day 17 - SVM, Naive Bayes and Clustering

**System 7 | Week 3 | Module B (Day 3)**
Topics: Naive Bayes vs SVM, K-Means, choosing k, hierarchical clustering, DBSCAN.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q9_kmeans_elbow_silhouette.py` | Q9 | Medium | K-Means for k = 2 to 10, elbow curve and silhouette scores to pick k |
| `q10_naive_bayes_vs_svm.py` | Q10 | Medium | Naive Bayes vs SVM on the same split, precision and recall compared |
| `bonus_hierarchical_dbscan.py` | Module B topic | | K-Means vs hierarchical vs DBSCAN on blobs and moons, plus a dendrogram |
| `outputs/` | | | Saved plots |

## How to run

```bash
pip install -r requirements.txt
mkdir -p outputs
python q9_kmeans_elbow_silhouette.py   # and so on
```

## Results

**Q9.** On synthetic data with 4 true groups, the silhouette score peaks at k = 4 (0.7741) and drops sharply after. The elbow curve bends at the same place, because inertia falls steeply up to k = 4 (from 150 at k = 3 to 43 at k = 4) and slowly afterwards. Plot: `outputs/q9_elbow_silhouette.png`.

**Q10.** Breast cancer dataset, same 80/20 split:

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Naive Bayes | 0.9386 | 0.9452 | 0.9583 |
| SVM (RBF) | 0.9737 | 0.9859 | 0.9722 |

The SVM wins on both. Naive Bayes assumes the features are independent of each other, and the cancer measurements (radius, perimeter, area) are strongly correlated, so that assumption hurts. The SVM does not make that assumption, and it needs feature scaling, which Naive Bayes does not.

**Bonus.** Adjusted Rand Index (ARI, 1.0 is perfect) against the true labels:

| Data | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Round blobs | 1.000 | 1.000 | 1.000 |
| Two moons | 0.475 | 0.482 | 0.995 |

Plots: `outputs/bonus_cluster_comparison.png`, `outputs/bonus_dendrogram.png`.

## Learning note: choosing k in K-Means

K-Means needs you to pick the number of clusters before it starts, so you need a way to choose.

- **Elbow method:** plot inertia (the total squared distance from points to their cluster centre) against k. It always falls as k grows, so look for the "elbow" where adding more clusters stops paying off. The weakness is that the elbow is often a judgement call.
- **Silhouette score:** for each point, compare how close it is to its own cluster against the nearest other cluster. It ranges from -1 to 1, and higher means tighter, better-separated clusters. Pick the k with the highest mean score. It is easier to read than the elbow, but slower on large data.
- **When DBSCAN is the better choice:** K-Means forces round, similar-sized clusters and puts every point in one. DBSCAN groups points by density, finds the number of clusters itself, handles odd shapes (the moons above) and labels outliers as noise. It is a better fit when clusters are not round or the data has outliers. Its weak spots are choosing `eps` and clusters with very different densities.

Analogy: K-Means is placing k tent poles and pulling every camper to the nearest pole, so the groups come out round. DBSCAN is walking through a crowd and joining anyone who stands close enough to someone already in the group, so a long curved queue still counts as one group.
