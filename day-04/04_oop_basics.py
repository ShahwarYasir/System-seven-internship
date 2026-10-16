class Dog:
    species = "Canis familiaris"            # class attribute

    def __init__(self, name, age):
        self.name = name                    # instance attributes
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"

    def __str__(self):
        return f"{self.name} ({self.age} years)"

class Puppy(Dog):                           # inheritance
    def bark(self):
        return f"{self.name} says yip!"

d = Dog("Rex", 5)
p = Puppy("Bubbles", 1)
print(d)                                    
print(d.bark())                            
print(p.bark())                            
print(isinstance(p, Dog), Dog.species)      
