# write
with open("notes.txt", "w") as f:
    f.write("Line one\n")
    f.write("Line two\n")

# append
with open("notes.txt", "a") as f:
    f.write("Line three\n")

# read whole file
with open("notes.txt", "r") as f:
    print(f.read())

# read line by line
with open("notes.txt") as f:
    for number, line in enumerate(f, start=1):
        print(number, line.strip())

# csv-style data
rows = [("Ali", 82), ("Sara", 91)]
with open("marks.csv", "w") as f:
    f.write("name,marks\n")
    for name, marks in rows:
        f.write(f"{name},{marks}\n")

with open("marks.csv") as f:
    header = f.readline().strip().split(",")
    print(header)
    for line in f:
        name, marks = line.strip().split(",")
        print(name, int(marks))
