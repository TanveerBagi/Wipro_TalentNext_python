#1. Write a program to accept two numbers and display their sum.
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))
print("Sum:", num1 + num2)

#2. Write a program to accept a welcome message and display the file name along with the welcome message.
welcome_message = input("Enter welcome message: ")
print("File name: simple_welcome.py")
print("Welcome message:", welcome_message)

# 3. Write a program to accept 10 numbers and calculate the sum of prime numbers among them.
count = 0
prime_sum = 0

while count < 10:
    num = int(input("Enter number {}: ".format(count + 1)))
    if num > 1:
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            prime_sum += num
    count += 1

print("Sum of prime numbers is:", prime_sum)
