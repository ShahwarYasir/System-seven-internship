student = {"name": "Ali", "age": 20, "cgpa": 3.4}

print(student["name"])              
print(student.get("city", "N/A"))   

student["city"] = "Islamabad"       # add
student["age"] = 21                 # update
del student["cgpa"]                 # delete
print(student)

# looping
for key, value in student.items():
    print(f"{key}: {value}")
print(list(student.keys()))
print(list(student.values()))

# nested dictionary
students = {
    101: {"name": "Ali", "marks": 82},
    102: {"name": "Sara", "marks": 91},
}
print(students[102]["name"])        # Sara

# dictionary comprehension
squares = {n: n ** 2 for n in range(1, 5)}
print(squares)                      

# counting with a dictionary
word = "banana"
counts = {}
for ch in word:
    counts[ch] = counts.get(ch, 0) + 1
print(counts)                      
