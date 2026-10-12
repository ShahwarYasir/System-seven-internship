# This is a single-line comment

"""
This is a multi-line string.
It is often used for longer notes or docstrings.
"""

# Indentation defines code blocks in Python
x = 10
if x > 5:
    print("x is greater than 5")   # indented = inside the if block

# Multiple assignment and None
p, q = 1, 2
nothing = None
print(p, q, nothing, type(nothing))   # 1 2 None <class 'NoneType'>

# Checking types
print(isinstance(x, int))             # True
print(type(3.0), type("hi"), type(True))