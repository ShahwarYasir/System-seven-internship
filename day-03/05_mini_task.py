# Word frequency counter and contact book using all four data structures
text = "the quick brown fox jumps over the lazy dog the end"
words = text.split()                          # list

freq = {}                                     # dict
for w in words:
    freq[w] = freq.get(w, 0) + 1

unique_words = set(words)                     # set
top = sorted(freq.items(), key=lambda item: item[1], reverse=True)[:3]

print("Total words:", len(words))
print("Unique words:", len(unique_words))
print("Top 3:", top)                         
# contact book: dict of name -> (phone, city) tuple
contacts = {
    "Ali": ("0300-1111111", "Lahore"),
    "Sara": ("0321-2222222", "Islamabad"),
}
contacts["Hamza"] = ("0333-3333333", "Karachi")
for name, (phone, city) in contacts.items():
    print(f"{name:6} {phone}  {city}")
