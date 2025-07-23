def ispalindrome(name):
    if name == name[::-1]:
        print("Yes, it is a palindrome.")
    else:
        print("No, it is not a palindrome.")

def count_the_vowels(name):
    vowels = 'aeiouAEIOU'
    count = sum(1 for char in name if char in vowels)
    print(f"No of vowels: {count}")

def frequency_of_letters(name):
    freq = {}
    for char in name:
        freq[char] = freq.get(char, 0) + 1
    print("Frequency of letters:")
    for k, v in freq.items():
        print(f"{k}-{v}", end=", ")
    print()
