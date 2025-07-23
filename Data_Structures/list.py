#1. Write a program to create a list of 5 integers and display the list items. Access individual elements through index.
numbers = [10, 25, 7, 42, 18]
print("Complete list:", numbers)

for i in range(len(numbers)):
    f"numbers[{i}] = {numbers[i]}"

#2. Write a program to append a new item to the end of the list.
nums = [1, 2, 3]
nums.append(4)
print(nums)

#3. Write a program to reverse the order of the items in the list.
nums2 = [1, 2, 3, 4]
nums_rev = nums2[::-1]

print(nums_rev)

#4. Print the number of occurrences of a specified element:
nums3 = [1, 2, 2, 3, 2]
count = 0
for i in range(len(nums3)):
    if nums3[i] == 2 :
        count+=1

print(f"Number of occurrences of 2 is {count}")

#5. Append items of list1 to the front of list2:
list1 = [1, 2]
list2 = [3, 4]
list2 = list1 + list2
print(list2)

#6. Insert a new item before the second element:
nums4 = [10, 20, 30]
nums4.insert(1, 15)
print(nums4)

#7. Remove item from a specified index:
nums5 = [10, 20, 30]
nums5.pop(2)
print(nums5)

#8. Remove first occurrence of a specified element:
nums6 = [10, 20, 30, 40, 30, 50]
nums6.remove(20)
print(nums6)
