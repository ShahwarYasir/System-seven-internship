"""Q14 (Easy): A single perceptron in NumPy that learns the AND gate.

Prediction : y_hat = step(w . x + b)
Learning   : w <- w + lr * (y - y_hat) * x ;  b <- b + lr * (y - y_hat)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# AND gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])


class Perceptron:
    def __init__(self, n_inputs, lr=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.1, n_inputs)
        self.b = 0.0
        self.lr = lr

    @staticmethod
    def step(z):
        return (z >= 0).astype(int)          # activation: fire if z >= 0

    def predict(self, X):
        return self.step(X @ self.w + self.b)

    def fit(self, X, y, max_epochs=100):
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.step(np.array(xi @ self.w + self.b))
                update = self.lr * (target - pred)
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            print(f"epoch {epoch:>2}: mistakes = {errors}  w = {np.round(self.w, 3)}  b = {self.b:.3f}")
            if errors == 0:                  # a full clean pass = converged
                print(f"\nConverged after {epoch} epochs.")
                return epoch
        print("\nDid not converge.")
        return max_epochs


p = Perceptron(n_inputs=2)
p.fit(X, y)

print("\nTruth table check:")
print(" x1 x2 | target | predicted")
for xi, t, pr in zip(X, y, p.predict(X)):
    print(f"  {xi[0]}  {xi[1]} |   {t}    |    {pr}")
assert (p.predict(X) == y).all(), "Perceptron failed to learn AND"

# Plot the learned decision boundary: w1*x1 + w2*x2 + b = 0
plt.figure(figsize=(4.5, 4.5))
for xi, t in zip(X, y):
    plt.scatter(*xi, c="tab:green" if t else "tab:red", s=140, edgecolors="k", zorder=3)
xs = np.linspace(-0.5, 1.5, 100)
if abs(p.w[1]) > 1e-9:
    plt.plot(xs, -(p.w[0] * xs + p.b) / p.w[1], "k--", label="decision boundary")
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Perceptron learns AND\n(green = 1, red = 0)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q14_perceptron_boundary.png", dpi=150)
print("Saved outputs/q14_perceptron_boundary.png")
