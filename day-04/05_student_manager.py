# Task 3: Student record manager that saves to a file
import json

FILENAME = "students.json"

def load_records():
    try:
        with open(FILENAME) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print("Data file is damaged, starting empty.")
        return {}

def save_records(records):
    with open(FILENAME, "w") as f:
        json.dump(records, f, indent=2)

def add_student(records):
    roll = input("Roll number: ").strip()
    if roll in records:
        print("That roll number already exists.")
        return
    name = input("Name: ").strip()
    try:
        marks = float(input("Marks (0-100): "))
        if not 0 <= marks <= 100:
            raise ValueError
    except ValueError:
        print("Marks must be a number between 0 and 100.")
        return
    records[roll] = {"name": name, "marks": marks}
    save_records(records)
    print("Saved.")

def view_students(records):
    if not records:
        print("No records yet.")
        return
    for roll, info in sorted(records.items()):
        print(f"{roll:8} {info['name']:15} {info['marks']}")

def delete_student(records):
    roll = input("Roll number to delete: ").strip()
    if records.pop(roll, None) is None:
        print("Not found.")
    else:
        save_records(records)
        print("Deleted.")

def main():
    records = load_records()
    while True:
        print("\n1. Add  2. View  3. Delete  4. Quit")
        choice = input("Choose: ").strip()
        if choice == "1":
            add_student(records)
        elif choice == "2":
            view_students(records)
        elif choice == "3":
            delete_student(records)
        elif choice == "4":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
