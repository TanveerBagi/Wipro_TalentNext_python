#Q1. Write a program to check if a given number is Positive, Negative or Zero.
Num = int(input("Enter the number: "))
if Num < 0:
    print("The number is Negative")
elif Num > 0:
    print("The number is Positive")
elif Num == 0:
    print("The number is Zero")

#Q2. Write a program to check if a given number is odd or even.
Num = int(input("Enter the number: "))
if Num % 2 == 0:
    print("The number is Even")
else :
    print("The number is odd")

#Q3. Given two non-negative values, print true if they have the same last digit, such as with 27 and 57

Num1 = int(input("Enter the First number: "))
Num2 = int(input("Enter the Second number: "))

if Num1 % 10 == Num2 % 10:
    print('true')
else:
    print('false')

#Q4. Write a program to print numbers from 1 to 10 in a single row with one tab space.
for i in range(1,11):
    print(i ,end = "\t")

#Q5. Write a program to print even numbers between 23 and 57. Each number should be printed in a seperate row.
for i in range(23, 58):
    if i % 2 == 0:
        print(i , end = "\n")

#Q6. Write a program to check if a given number is prime or not.
Num = int(input("Enter the number: "))
is_prime = True
for i in range(2, (Num // 2)+1):
    if Num % i == 0:
        is_prime = False

    else:
        continue

if not is_prime :
    print(f"{Num} is not a prime number")
else:
    print(f"{Num} is a prime number")

#Q7. Write a program to print prime numbers between 10 and 99.
for i in range(10, 100):
    is_prime = True
    for j in range(2, (i // 2) + 1):
        if i % j == 0:
            is_prime = False
            break

    if is_prime:
        print(i)

#Q8. Write a program to print the sum of all the digits of a given number.
Num = int(input("Enter the number: "))
Sum = 0
while Num!= 0:
    Sum += Num % 10
    Num = Num // 10

print(Sum)

#Q9. Write a program to reverse a given number and print.
Num = int(input("Enter the number: "))
Rev_Num = 0
while Num!= 0:
    Rev_Num = (Rev_Num * 10) + (Num % 10)
    Num = Num // 10

print(Rev_Num)

#Q10. Write a program to find if the given number is palindrome or not.
Num = int(input("Enter the number: "))
Rev_Num = 0
while Num!= 0:
    Rev_Num = (Rev_Num * 10) + (Num % 10)
    Num = Num // 10

if Num == Rev_Num:
    print(f"{Num} is a palindrome")

else:
    print(f"{Num} is not a palindrome")
