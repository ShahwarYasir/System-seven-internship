fruits = ["apple", "banana", "cherry"]

# access and slicing
print(fruits[0], fruits[-1])        
print(fruits[0:2])                 

# modify
fruits.append("mango")
fruits.insert(1, "orange")
fruits.remove("banana")
print(fruits)                      
print(fruits.pop())                 
print(len(fruits))                 

# sorting
nums = [5, 2, 9, 1]
nums.sort()
print(nums)                         
print(sorted(nums, reverse=True))   

# list comprehension
squares = [n ** 2 for n in range(1, 6)]
evens = [n for n in range(1, 11) if n % 2 == 0]
print(squares)                      
print(evens)                        

# looping
for i, f in enumerate(fruits):
    print(i, f)
