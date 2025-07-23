#1. Write a program to remove a given item from the set.
set1 = {1,2,3,4,5}
set1.remove(2)
print(set1)

#2. Write a program to create an intersection of sets.
set2 = {1,2,3,4,5}
set3 = {3,4,5,6,7}
common = set2 & set3
print(common)

#3. Write a program to create an union of sets.
set4 ={1,2,3,4,5}
set5 = {3,2,4,5,7,8}
union = set4 | set5
print(union)

#4. Write a program to find the maximum and minimum value in a set.
set7 = {3,2,4,5,7,8}
maximum = max(set7)
minimum = min(set7)
print(f"maxumum: {maximum}, minimum{minimum}")
