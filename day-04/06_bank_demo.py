# Task 4: Bank Account class (module: bank.py)
from bank import BankAccount, SavingsAccount, InsufficientFunds

acc = BankAccount("Ali", 1000)
acc.deposit(500)
acc.withdraw(200)
print(acc)                          # Ali: Rs. 1,300.00

try:
    acc.withdraw(5000)
except InsufficientFunds as e:
    print("Failed:", e)

sav = SavingsAccount("Sara", 10000, rate=0.10)
sav.add_interest()
print(sav)                          # Sara: Rs. 11,000.00
print(sav.history)
