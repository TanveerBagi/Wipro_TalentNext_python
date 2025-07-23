#1. Write a program to print the 4th element from first and 4th element from last in a tuple.
tupl = (1,2,3,4,5)
print(f"4th element from first: {tupl[3]}")
print(f"4th element from last: {tupl[-4]}")

#2. Write a program to check whether an element exists in a tuple or not.
tupl2 = (1,4,3,6,9)
num = 3
exists = False
for i in range(len(tupl2)):
    if tupl2[i] == num:
        print("element exists in tuple")
        exists = True

if not exists:
    print("Element does not exist")

#3. Write a program to convert a list into a tuple.
list1 = [1,9,5,4,5,4,8]
tupl4 = tuple(list1)
print(tupl4)

#4.Write a program to find the index of an item in a tuple.
tupl5 = (10, 20, 40, 55, 14, 44, 85)
for  i in range(len(tupl5)):
    if tupl5[i] == 40:
        print(f"the of the 40 is {i}")

#5. Write a program to replace last value of tuples in a list to 100.
sam_list = [(10, 20, 40), (40, 50, 60), (70, 80, 90)]
result = []
for i in sam_list:
    i = list(i)
    i[-1] = 100
    result.append(tuple(i))
print(result)
