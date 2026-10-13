def calculate(a, op, b):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            return "Error: division by zero"
        return a / b
    else:
        return "Error: unknown operator"

while True:
    choice = input("Press Enter to calculate or q to quit: ")
    if choice.lower() == "q":
        break
    a = float(input("First number: "))
    op = input("Operator (+, -, *, /): ")
    b = float(input("Second number: "))
    print("Result:", calculate(a, op, b))