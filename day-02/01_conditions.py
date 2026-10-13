marks = 72

if marks >= 80:
    grade = "A"
elif marks >= 70:
    grade = "B"
elif marks >= 60:
    grade = "C"
else:
    grade = "F"
print(f"Marks {marks} -> Grade {grade}")   

num = 7
print("even" if num % 2 == 0 else "odd")   

age = 21
if age >= 18 and age < 65:
    print("working-age adult")             