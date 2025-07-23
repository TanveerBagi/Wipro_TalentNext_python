#1.Write a program to return the sum of all the numbers in a list
def summ_of_list(list1):
    summ = 0
    for i in range(len(list1)):
        summ += list1[i]

    return summ
list1 = [1,2,3,4,5]
print(summ_of_list(list1))

#2. Write a function to return the reverse of a string
def rev_str(string1):
    reverse_string =string1[::-1]
    return reverse_string
string1 ="ABCDEF"
print(rev_str(string1))

#3. Write a function to calculate and return the factorial of a number
def fact(num):
    factorial_of_num =1
    for i in range(1, num+1):
        factorial_of_num *= i

    return factorial_of_num
print(fact(3))

#4. Write a function that accepts a string and prints the number of upper case letters and lower case letters in it.
def upper_lower_case(string):
    is_upper = 0
    is_lower = 0
    for i in range(len(string)):
        if string[i].isupper():
            is_upper +=1

        elif string[i].islower():
            is_lower +=1

    return is_lower , is_upper

string2 = "HelloWorld"
print(f"The Number of lowercase and uppercase letters in the given string are {upper_lower_case(string2)}")

#4. Write a function to print the even numbers from a given list. List is passed to the function as an argument.
def is_even(list11):
    even_num = []
    for i in list11:
        if i % 2 == 0:
            even_num.append(i)

        else:
            continue
    return even_num


print(is_even([1, 2, 3, 4, 5, 6, 7, 8, 9]))

#5. Write a function that takes a number as a parameter and checks whether the number is prime or not.
def num_is_prime(num):
    if num ==0:
        return "The given number is zero"
    is_prime = True
    for i in range(2,int(num/2+1)):
        if num % i ==0:
            is_prime = False

        else:
            continue

    if is_prime:
        print("The number is prime")
    elif not is_prime:
        print("The number is not prime")

num_is_prime(3)
