# Simple CGPA-style calculator
credit1 = float(input("Credit hours for course 1: "))
gpa1 = float(input("GPA for course 1: "))
credit2 = float(input("Credit hours for course 2: "))
gpa2 = float(input("GPA for course 2: "))

total_points = credit1 * gpa1 + credit2 * gpa2
total_credits = credit1 + credit2
print(f"Semester GPA: {total_points / total_credits:.2f}")