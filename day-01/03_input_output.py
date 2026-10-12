name = input("Enter your name: ")
marks = float(input("Enter your marks out of 100: "))
percent_of_total = marks / 100 * 40   # weight of 40

print(f"Hi {name}, your marks are {marks}")
print(f"Weighted score: {percent_of_total:.2f} / 40")