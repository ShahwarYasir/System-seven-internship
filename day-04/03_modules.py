import math
import random
from datetime import datetime

import mymath                       # our own module (mymath.py)
from mymath import circle_area

print(math.sqrt(49), math.ceil(4.2), math.floor(4.8))   # 7.0 5 4
print(random.randint(1, 6))
print(random.choice(["red", "green", "blue"]))
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

print(mymath.is_even(10))           # True
print(round(circle_area(3), 2))     # 28.27
