#1. Write a LC program to create an output dictionary which contains only the odd numbers that are present in the input list = [1,2,3,4,5,6,7] as keys and their cubes as values

input_list = [1,2,3,4,5,6,7]
odd_cubes_dict = {x: x**3 for x in input_list if x % 2 != 0}
print(odd_cubes_dict)

#2. Make a dictionary of the 26 english alphabets mapping each with the corresponding integer

alphabet_dict = {chr(i): i-96 for i in range(97, 123)}
print(alphabet_dict)
