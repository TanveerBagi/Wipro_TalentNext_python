#1.Create a dictionary that contains a list of people and one interesting fact about each of them.
  #Display each person and his or her interesting fact to the screen. Next, change a fact about one of the people. Also add an additional person and corresponding fact. Display the new list of people and facts. Run the program multiple times and notice if the order changes.

people_and_fact = {
   "Jeff": "Is afraid of Dogs.",
   "David": "Plays the piano.",
   "Jason": "Can fly an airplane"
}
for key , values in people_and_fact.items():
    print(key ,":", values)

print("\n")

people_and_fact["Jeff"] = "Loves cats"

people_and_fact["Jill"] = "Can hula dance"

for key , values in people_and_fact.items():
    print(key ,":", values)

#2. Given the participant's score sheet for the university sports day, you are required to find the runner up score. Store them in a list and find the score for the runner-up
Scores = [2,3,6,6,5]
score_of_first = 1
score_of_runnerup = 1
Scores.sort()
for i in range(len(Scores)):
    if Scores[i]> score_of_first:
        score_of_runnerup =score_of_first
        score_of_first = Scores[i]

print(f"Score of the runnerup is: {score_of_runnerup}")

#3.You have a record of n students. Each record contains the student's name and their percent marks in Math, Physics, and Chemistry. The marks can be floating values. You are required to save the record in a dictionary data type.
   #The student’s name is the key. Marks stored in a list is the value. The user enters a student’s name. Output the average percentage marks obtained by that student.
Record = {
    "krishna" : [67,68,69],
    "Arjun" :[70,98,63],
    "Malika" :[52,56,60]

}

name_of_sudent = input("Enter the name: ")
total = 0
for key, values in Record.items():
    if key == name_of_sudent:
        for j in range(len(values)):
            total+= values[j]

    else:
        continue

avg_percent_marks = total / 3
print(f"The average percentage marks of {name_of_sudent} is {avg_percent_marks}")

#4. Given a string of n words, help ALEX to find out how many times his name appears in the string
name ="Alex"
string_of_words ="Hi Alex Welcome Alex Bye Alex"
count = 0
for j in string_of_words.split():
    if j == name:
        count += 1
print(count)

