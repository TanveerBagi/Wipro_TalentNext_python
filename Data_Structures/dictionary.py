#1. Write a program to add a key and value to a dictionary.
Dictionary = {0: 10, 1: 20}
Dictionary[2] = 30
print(Dictionary)

#2. Write a program to concatenate the following dictionaries to create a new one.
dic1 = {1: 10, 2: 20}
dic2 = {3: 30, 4: 40}
dic3 = {5: 50, 6: 60}
result = {}
result.update(dic1)
result.update(dic2)
result.update(dic3)
print(result)

#3. Write a program to check if a given key already exists in a dictionary.
d = {1: 'a', 2: 'b'}
key = 2
if key in d:
    print("Key exists")
else:
    print("Key does not exist")

#4.Write a program to iterate over a dictionary using for loop and print the keys alone, values alone and both keys and values.
d = {1: 'a', 2: 'b'}

print("keys: ")
for keys in d:
    print(keys)

print("Values: ")
for values in d.values():
    print(values)

print("Both: ")
for keys, values in d.items():
    print(keys, values)

#5. Write a program to prepare a dictionary where the keys are numbers between 1 and 15 (both included) and the values are square of the keys.
dict = {}
for i in range(1,16):
    dict[i] = i*i

print(dict)

#6. Write a program to sum all the values in a dictionary, considering the values will be of int type.
d = {1: 10, 2: 20, 3: 30}
total = 0
for i in d.values():
    total+= i

print(total)
