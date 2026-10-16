# basic try / except
try:
    number = int("abc")
except ValueError:
    print("That is not a valid number")

# multiple handlers, else and finally
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    except TypeError:
        print("Both values must be numbers")
        return None
    else:
        return result
    finally:
        print("division attempted")

print(safe_divide(10, 2))           
print(safe_divide(10, 0))           
print(safe_divide(10, "x"))         

# file not found
try:
    with open("missing.txt") as f:
        f.read()
except FileNotFoundError:
    print("File does not exist")

# raising your own exception
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

try:
    set_age(-5)
except ValueError as e:
    print("Error:", e)

# input validation loop
while True:
    try:
        value = int(input("Enter an integer: "))
        break
    except ValueError:
        print("Try again")
print("You entered", value)
