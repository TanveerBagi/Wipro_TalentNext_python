import mymodule   #for miniproject 2

#1.Write a Python function that accepts a hyphen-separated sequence of colors as input and returns the colors in a hyphen-separated sequence after sorting them alphabetically.
def sort_colors(input_string):
    colors = input_string.split("-")
    colors.sort()
    return "-".join(colors)

input1 = "green-red-yellow-black-white"
output1 = sort_colors(input1)
print("Output 1:", output1)

#2. Write a module called mymodule.py that contains the following functions:
  #ispalindrome(name)
  #→ This function should check whether the given name is a palindrome or not.
  #→ If it is, print "Yes, it is a palindrome.", otherwise print "No, it is not a palindrome.".
  #count_the_vowels(name)
  #→ This function should count and print the number of vowels present in the given name.
  #→ Output format: "No of vowels: X"
  #frequency_of_letters(name)
  #→ This function should calculate and display the frequency of each letter in the name.
  #→ Output format example: "Frequency of letters: b-2, o-1"

# The code Functions are in string_utils.py file
name = "bob"
mymodule.ispalindrome(name)
mymodule.count_the_vowels(name)
mymodule.frequency_of_letters(name)
