"""Q6 (Medium): Train vs test score for max_depth 1..15 to show overfitting."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

depths = range(1, 16)
train_scores, test_scores = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

best_depth = depths[test_scores.index(max(test_scores))]
print(f"{'depth':>5} {'train':>8} {'test':>8}")
for d, tr, te in zip(depths, train_scores, test_scores):
    print(f"{d:>5} {tr:>8.4f} {te:>8.4f}")
print(f"\nBest test accuracy {max(test_scores):.4f} at max_depth={best_depth}")

plt.figure(figsize=(7, 4.5))
plt.plot(depths, train_scores, "o-", label="Training score")
plt.plot(depths, test_scores, "s-", label="Test score")
plt.axvline(best_depth, color="gray", ls="--", label=f"Best depth = {best_depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Overfitting in a decision tree")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q6_overfitting_curve.png", dpi=150)
print("Saved outputs/q6_overfitting_curve.png")
