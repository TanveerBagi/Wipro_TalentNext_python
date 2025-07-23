#1. Write a program to count the number of upper and lower case letters in a String.
string1 = "HelloWorld"
upper_case =0
lower_case =0
for i in range(len(string1)):
    if string1[i].isupper():
        upper_case+=1

    elif string1[i].islower():
        lower_case+=1

print(f"Number of uppercase letters are: {upper_case}, number of lowercase letters are: {lower_case}")

#2. Write a program that will check whether a given String is Palindrome or not.
string2 = "Racecar"
string2 = string2.lower()
rev = string2[::-1]
if rev ==string2:
    print("String is palindrome")

else:
    print("Not palindrome")

#3. Given a string, return a new string made of n copies of the first 2 chars of the original string where n is the length of the string. The string length will be >= 2.
    #If input is "Wipro" then output should be "WiWiWiWiWi".
string3 ="Wipro"
n = len(string3)
two_characters =""
two_characters = string3[:2]

new_string =two_characters * n
print(new_string)

#4. Given a string, if the first or last character is 'x', return the string without those 'x' characters, else return the string unchanged.
    #If the input is "xHix", then output is "Hi".

string4 ="xHix"
if string4[1] == "x":
    string4 = string4.replace(string4[1],"")

elif string4[-1] =="x":
    string4 = string4.replace(string4[-1],"")

print(string4)

#5.Given a string and an integer n, return a string made of n repetitions of the last n characters of the string. You may assume that n is between 0 and the length of the string inclusive.
string5 = "Wipro"
n = 3
last_n = len(string5) - n
last_n_characters = string5[last_n:]

new_string5 = last_n_characters * n
print(new_string5)
