"""A small module that other scripts import."""

PI = 3.14159

def circle_area(radius):
    return PI * radius ** 2

def is_even(n):
    return n % 2 == 0

if __name__ == "__main__":
    # runs only when executed directly, not when imported
    print("Testing mymath:", circle_area(2), is_even(4))
