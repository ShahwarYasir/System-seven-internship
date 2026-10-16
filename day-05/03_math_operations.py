import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print(a + b, a - b, a * b, a / b)
print(a ** 2)
print(np.sqrt(a))

data = np.array([[2, 4, 6], [1, 3, 5]])
print(data.sum(), data.mean(), data.max(), data.min())
print(data.sum(axis=0))             # column sums [3 7 11]
print(data.sum(axis=1))             # row sums    [12 9]
print(np.std(data), np.median(data))

# broadcasting: a scalar or smaller array is stretched to fit
print(data + 10)
print(data * np.array([1, 10, 100]))

# dot product and matrix multiplication
print(np.dot(b, b))                 # 30
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)                        # [[19 22] [43 50]]
print(A.T)                          # transpose

# random numbers (seed makes them repeatable)
rng = np.random.default_rng(42)
print(rng.integers(1, 100, size=5))
