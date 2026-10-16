import numpy as np

m = np.arange(1, 13).reshape(3, 4)
print(m)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

print(m[0, 0])          # 1
print(m[2, -1])         # 12
print(m[1])             # [5 6 7 8]        row 1
print(m[:, 2])          # [ 3  7 11]       column 2
print(m[0:2, 1:3])      # [[2 3] [6 7]]    sub-matrix

# boolean masks
print(m[m > 6])         # [ 7  8  9 10 11 12]
print(m[m % 2 == 0])    # [ 2  4  6  8 10 12]

# modifying through a mask
m2 = m.copy()
m2[m2 > 10] = 0
print(m2)

# fancy indexing
print(m[[0, 2], :])     # rows 0 and 2
