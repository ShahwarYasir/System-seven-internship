def square(n):
    return n * n

def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

def min_max(numbers):
    return min(numbers), max(numbers)

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print(square(6))                    
print(greet("Shahwar"))             
print(greet("Shahwar", "Salam"))    
print(min_max([4, 9, 1, 7]))       
print(factorial(5))                