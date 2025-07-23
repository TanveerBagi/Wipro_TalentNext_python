"""
1.Through command line arguments, three strings separated by space are given to you.
Each string is a series of numbers separated by hyphen (-). You like all the numbers in string 1. You dislike all the numbers in string 2. String 3 contains the numbers given to you
You dislike all the numbers in string 2. 

String 3 contains the numbers given to you. Your initial happiness is 0. For each number in string 3. If it’s in string 1 → add 1
If it’s in string 2 → subtract 1, Else → no change
Output your final happiness.
"""

liked = input("Enter liked numbers (hyphen-separated): ").split('-')
disliked = input("Enter disliked numbers (hyphen-separated): ").split('-')
given = input("Enter given numbers (hyphen-separated): ").split('-')

happiness = 0

for num in given:
    if num in liked:
        happiness += 1
    elif num in disliked:
        happiness -= 1

print("Final happiness:", happiness)
liked = input("Enter liked numbers (hyphen-separated): ").split('-')
disliked = input("Enter disliked numbers (hyphen-separated): ").split('-')
given = input("Enter given numbers (hyphen-separated): ").split('-')

happiness = 0

for num in given:
    if num in liked:
        happiness += 1
    elif num in disliked:
        happiness -= 1

print("Final happiness:", happiness)
