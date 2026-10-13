for i in range(5):
    print(i)                 

print(list(range(2, 10, 2)))  # [2, 4, 6, 8]
print(list(range(5, 0, -1)))  # [5, 4, 3, 2, 1]

# while loop: sum 1 to 10
total, n = 0, 1
while n <= 10:
    total += n
    n += 1
print(total)                  # 55

# break and continue
for i in range(1, 11):
    if i % 2 == 0:
        continue              # skip even numbers
    if i > 7:
        break                 # stop after 7
    print(i)                  # 1 3 5 7

# multiplication table
for i in range(1, 6):
    print(f"5 x {i} = {5 * i}")