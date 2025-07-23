'''1. create a python program that asks the user how far they want to travel. if they want to travel less than three miles tell them to ride Bicycle. if they want to travel more than three miles, but less than three hundred miles, tell them to ride Motor-cycle. if they want to travel three hundred miles or more tell them to driver Super-Car.
Sample Output:
          How far would you like to travel in miles? 2500
          I suggest Super-Car to your destination'''

miles_to_go = int(input("How far would you travel in miles: "))
if miles_to_go < 3:
    print("I suggest you ride bicycle to your destination")
elif 3 < miles_to_go < 300:
    print("I suggest you ride motor-cycle to your destination")
elif miles_to_go >= 300:
    print("I suggest Super-Car to your destination")
else:
    print("Please enter a valid input")

'''2. Let's assume you are planning to use your python skills to build an App for Mobile.
You decide to host your application on servers running in the cloud. you pick a hosting provider that charges $0.51 per hour. you will launch your services using one server and want to know how much it will cost to operate per day, per week, per month.
Write a python program that displays the answers to the following questions:
How much does it cost to operate one server per day?
How much does it cost to operate one server per week?
How much does it cost to operate one server per month?
How much days can I operate one server with $918?'''

per_hour_cost = 0.51
per_day_cost = per_hour_cost * 24
per_week_cost = per_day_cost * 7
per_month_cost = per_day_cost * 30
days_with_918 = int(918 / per_day_cost)

print(f"It will cost ${per_day_cost} to operate one server per day.")
print(f"It will cost ${per_week_cost} to operate one server per week.")
print(f"It will cost ${per_month_cost} to operate one server per month.")  # Assuming the month has 30 days
print(f"You can operate one server with $918 for {days_with_918} days.")
