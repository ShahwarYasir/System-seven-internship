a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)                        # union        {1, 2, 3, 4, 5, 6}
print(a & b)                        # intersection {3, 4}
print(a - b)                        # difference   {1, 2}
print(a ^ b)                        # symmetric    {1, 2, 5, 6}

# sets remove duplicates
data = [1, 2, 2, 3, 3, 3]
print(set(data))                    

a.add(10)
a.discard(1)
print(a)
print(3 in a, 99 in a)             

# empty set: {} makes a dict, so use set()
empty = set()
print(type({}), type(empty))
