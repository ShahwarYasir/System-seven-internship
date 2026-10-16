import numpy as np

a = np.array([1, 2, 3, 4])
m = np.array([[1, 2, 3], [4, 5, 6]])

print(a, a.dtype)                   # [1 2 3 4] int64
print(m.shape, m.ndim, m.size)      # (2, 3) 2 6

print(np.zeros((2, 3)))
print(np.ones((2, 2)))
print(np.arange(0, 10, 2))          # [0 2 4 6 8]
print(np.linspace(0, 1, 5))         # [0.   0.25 0.5  0.75 1.  ]
print(np.eye(3))                    # identity matrix

r = np.arange(12).reshape(3, 4)
print(r)

# why NumPy: vectorised math, no loops
print(a * 2)                        # [2 4 6 8]
print([x * 2 for x in [1, 2, 3, 4]])  # same result with a list comprehension
