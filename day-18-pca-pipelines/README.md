# Day 18 - PCA, Imbalanced Data, Leakage and Pipelines

**System 7 | Week 3 | Module B (Day 4)**
Topics: class weights, PCA, pipelines, data leakage, feature engineering, customer segmentation task.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q11_class_weight_imbalance.py` | Q11 | Medium | 95/5 imbalanced data, recall before and after `class_weight` |
| `q12_pca_clusters.py` | Q12 | Hard | PCA to 2 components, cluster plot, explained variance |
| `q13_pipeline_and_leakage.py` | Q13 | Hard | Scaler, PCA and classifier pipeline, plus leakage demos |
| `segmentation_task.py` | Practical task | | Customer segmentation with feature engineering, PCA and K-Means |
| `outputs/` | | | Plots and `segment_profiles.csv` |

## How to run

```bash
pip install -r requirements.txt
mkdir -p outputs
python q11_class_weight_imbalance.py   # and so on
python segmentation_task.py
```

## Results

**Q11.** Data is 95% negative and 5% positive. Recall for the rare class before and after `class_weight="balanced"`:

| Model | class_weight | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | None | 0.000 | 0.000 | 0.000 |
| Logistic Regression | balanced | 0.134 | 0.765 | 0.228 |
| Random Forest | None | 1.000 | 0.147 | 0.256 |
| Random Forest | balanced | 1.000 | 0.206 | 0.342 |

Without weights, logistic regression never predicts the rare class at all, yet its accuracy is 0.943, which is almost the share of the majority class. That is why accuracy misleads here. Weighting lifts recall from 0 to 0.765, but accuracy falls to 0.718 and precision is low, so there are many false alarms. It is a trade-off: it is worth it when a miss costs more than a false alarm. The random forest improves much less with weights alone, so it would need other techniques such as resampling or threshold tuning. Plot: `outputs/q11_class_weight_confusion.png`.

**Q12.** Wine data, 13 features reduced to 2 components. PC1 explains 36.20% and PC2 19.21% of the variance, 55.41% together, so 44.59% of the information is lost in the 2D view. K-Means on those two components still recovers the three wine classes well (ARI 0.895). PC1 is driven mainly by flavanoids and total phenols, PC2 by colour intensity, alcohol and proline. Reaching 95% of the variance takes 10 of the 13 components. Plot: `outputs/q12_pca_clusters.png`.

**Q13.**
- **Part A:** the pipeline (scaler, PCA with 10 components, logistic regression) scores 0.9737 on the test set and keeps 95.27% of the variance.
- **Part B (scaling before the split, as asked):** over 50 random splits the leaky and correct versions score the same (0.9793 vs 0.9795). Scaling leakage is real but small, because only a mean and a standard deviation leak, not labels. So this is bad practice that rarely changes the score, not proof that leakage is harmless.
- **Part C (a stronger demo):** 5,000 random features and random labels, so the true accuracy is about 0.50. Selecting the "best" 20 features before cross-validation gives 0.85, which is pure leakage. Doing the selection inside a pipeline gives 0.48, the honest answer.

### Practical task: customer segmentation

600 synthetic customers built from four hidden types. Features: age, income, visits per year, average basket, plus two engineered ones, `annual_spend` (visits x basket) and `spend_pct_of_income`. Steps: scale, PCA (keep 90% variance, which gives 2 components), K-Means with k chosen by silhouette (k = 4, score 0.743).

| Segment | Customers | Age | Income (k) | Visits/yr | Avg basket | Annual spend | Spend % of income | Suggested name |
|---|---|---|---|---|---|---|---|---|
| 0 | 139 | 23.9 | 45.0 | 55.0 | 39.6 | 2,167 | 4.9 | Young trendsetters: frequent, mid basket, spend the most relative to income |
| 1 | 120 | 41.0 | 109.7 | 25.2 | 95.0 | 2,388 | 2.2 | Big spenders: high income, large baskets |
| 2 | 191 | 33.3 | 35.8 | 39.8 | 18.6 | 737 | 2.1 | Budget regulars: frequent but small baskets |
| 3 | 150 | 51.1 | 69.6 | 6.0 | 60.5 | 363 | 0.5 | Occasional shoppers: rare visits, an opportunity for win-back offers |

The segments match the four hidden types almost exactly (only 11 young-trendy customers fall into the budget-regular segment). That is expected, since the data is synthetic and well separated, so real customer data would be messier. Plots and tables: `outputs/segmentation.png`, `outputs/segment_profiles.csv`.

## Learning notes

### Data leakage

Leakage is when information from outside the training data, usually from the test set, influences how the model is built, so the score looks better than real-world performance will be. It happens whenever a step that learns from data (scaling, imputing, PCA, feature selection) is fitted on the whole dataset before splitting. The test rows then help shape the preprocessing.

The fix is to split first and wrap every learning step in a `Pipeline`, so each step is refitted on only the training part (inside each fold during cross-validation). Part C of Q13 shows how bad it can get: a leaky workflow reported 85% accuracy on data with nothing to learn.

Analogy: an exam where you saw the answer key while choosing what to study. Your score looks great, but it says nothing about how you will do on questions you have not seen.

### Curse of dimensionality

As the number of features grows, the data becomes thin. Space grows exponentially, so points sit far apart, distances become less meaningful and models need far more samples to avoid fitting noise. More features can mean worse generalization. The wine data shows a milder version: 13 features can be compressed to 10 components with almost no loss, so many of them carry overlapping information.

- **PCA** helps by keeping the directions with the most variance and dropping the rest, which gives fewer, uncorrelated inputs.
- **Feature selection** helps by dropping features that carry little signal. Do this inside the pipeline, or it causes the leakage above.

Analogy: 100 people spread along a corridor are easy to find. Spread the same 100 people through a huge building with many floors and they rarely meet, so you can learn very little about any one of them from their neighbours.
