#1. Write a program to accept two numbers from the user and perform division. If any exception occurs, print an error message or else print the result.
try:
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))
    result = a / b
    print("Result of division is:", result)
except Exception as e:
    print("Error:", e)

#2. Write a program to accept a number from the user and check whether it’s prime or not. If user enters anything other than number, handle the exception and print an error message.
try:
    num = int(input("Enter a number: "))
    if num <= 1:
        print("Not a prime number")
    else:
        for i in range(2, num):
            if num % i == 0:
                print("Not a prime number")
                break
        else:
            print("Prime number")
except:
    print("Invalid input. Please enter a number.")

#3. Write a program to accept the file name to be opened from the user, if file exists print the contents of the file in title case or else handle the exception and print an error message.
try:
    filename = input("Enter file name: ")
    with open(filename, "r") as file:
        content = file.read()
        print(content.title())
except Exception as e:
    print("Error:", e)

#4.Declare a list with 10 integers and ask the user to enter an index. Check whether the number in that index is positive or negative number. If any invalid index is entered, handle the exception and print an error message.
numbers = [10, -5, 7, -3, 15, -9, 0, 8, -2, 4]

try:
    index = int(input("Enter an index (0-9): "))
    value = numbers[index]
    if value > 0:
        print("Positive number")
    elif value < 0:
        print("Negative number")
    else:
        print("Zero")
except Exception as e:
    print("Error:", e)
