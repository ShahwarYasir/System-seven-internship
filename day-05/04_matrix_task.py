# Task 5: NumPy operations on a sample matrix
import numpy as np

rng = np.random.default_rng(7)
matrix = rng.integers(1, 50, size=(5, 5))
print("Matrix:\n", matrix)

print("\nShape:", matrix.shape)
print("Sum:", matrix.sum(), "| Mean:", round(matrix.mean(), 2))
print("Row sums:   ", matrix.sum(axis=1))
print("Column means:", matrix.mean(axis=0).round(2))

print("\nDiagonal:", np.diag(matrix))
print("Trace:", np.trace(matrix))
print("Upper-left 3x3:\n", matrix[:3, :3])
print("Values above 30:", matrix[matrix > 30])

# replace values below the mean with 0
filtered = np.where(matrix >= matrix.mean(), matrix, 0)
print("\nBelow-mean values zeroed:\n", filtered)

# normalise to the 0-1 range
norm = (matrix - matrix.min()) / (matrix.max() - matrix.min())
print("\nNormalised:\n", norm.round(2))

# matrix algebra
print("\nTranspose:\n", matrix.T)
print("Matrix x its transpose:\n", matrix @ matrix.T)
print("Determinant:", round(np.linalg.det(matrix), 2))
