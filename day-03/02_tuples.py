point = (3, 4)
print(point[0], point[1])           


try:
    point[0] = 10
except TypeError as e:
    print("Error:", e)

# unpacking
x, y = point
print(x, y)                         

# a single-item tuple needs a trailing comma
single = (5,)
print(type(single), type((5)))      

# functions returning tuples
def min_max(values):
    return min(values), max(values)

low, high = min_max([4, 8, 1, 9])
print(low, high)                    

# swapping with tuples
a, b = 1, 2
a, b = b, a
print(a, b)                         
