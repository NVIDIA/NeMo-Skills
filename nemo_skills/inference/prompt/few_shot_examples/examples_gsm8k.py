# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

text_with_code = [
    {
        "question": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
        "reference_solution": "We let x be the amount of money she put in the bank the first year.\nThen the second year was 2*x, \nand the third year was 2*2*x=4*x,\nand the fourth year was 2*2*2*x=8*x.\nAnd the total amount she put in the bank, $450, is the sum of all four years, or $450 = x+(2*x)+(4*x)+(8*x).\nThis simplifies to $450=15*x.\nAnd therefore, the amount of money she put in the bank the first year was x=$30.",
        "masked_reference_solution": "We let x be the amount of money she put in the bank the first year,\nThen the second year was 2*x,\nand the third year was 2*2*x=M*x,\nand the fourth year was 2*2*2*x=N*x.\nAnd the total amount she put in the bank, $450, is the sum of all four years, or $450 = x+(2*x)+(M*x)+(N*x).\nThis simplifies to $450=(1 + 2 + M + N)*x.\nAnd therefore, the amount of money she put in the bank the first year was x=P dollars.",
        "generated_solution": """
Missy started with some amount of money in the first year.
She was then doubling the amount of money every year for 4 years and ended up with $450.
Let's write down an equation for this problem and solve it using sympy.
{start_code}
from sympy import solve, symbols
first_year_money = symbols('first_year_money')
second_year_money = 2 * first_year_money
third_year_money = 2 * second_year_money
fourth_year_money = 2 * third_year_money
# Solve for first year money given that the combined saving is 450
result = solve(first_year_money + second_year_money + third_year_money + fourth_year_money - 450, first_year_money)[0]
result
{end_code}
{start_code_output}
30
{end_code_output}
Thus Missy put \\boxed{30} dollars in the bank the first year.
""".strip(),
    },
    {
        "question": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = (60+20) minutes = 80 minutes\nHe will spend a total of 80+10 = 90 minutes\n90 minutes is = (60/60) hours and 30 minutes = 1 hour 30 minutes = 0130 in 24-hr time\nHe has to leave 0130 hours earlier than 0900 i.e. 0900-0130 = 0730 hours",
        "masked_reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = M minutes\nHe will spend a total of M+10 = N minutes\nN minutes is = O hours and P minutes = O hour P minutes = Q in 24-hr time\nHe has to leave Q hours earlier than 0900 i.e. 0900-Q = R hours",
        "generated_solution": """
Since Pete needs to take a 10 minutes walk and then a 1 hour 20 minutes train ride, he will spend a total of 1 hour and 30 minutes.
This means that he needs to leave 1 hour and 30 minutes earlier than 09:00 hours.
Subtracting 1 hour and 30 minutes from 09:00 hours we get \\boxed{07:30} hours.
""".strip(),
    },
    {
        "question": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "reference_solution": "Five times the amount Mark deposited is $88 x 5 = $440.\nSo, Bryan deposited $440 - $40 = $400.",
        "masked_reference_solution": "Five times the amount Mark deposited is $88 x 5 = $M.\nSo, Mark deposited $M - $40 = $N.",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
mark_deposit = 88
five_times_mark_deposit = 5 * mark_deposit
bryan_deposit = five_times_mark_deposit - 40
bryan_deposit
{end_code}
{start_code_output}
400
{end_code_output}
So Bryan deposited \\boxed{400} dollars in the bank.
""".strip(),
    },
    {
        "question": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "reference_solution": "The amount of the discount is $120 x 0.2 = $24.\nThe new price will be $120 \u2013 $24 = $96.",
        "masked_reference_solution": "The amount of the discount is $120 x 0.2 = $M.\nThe new price will be $120 \u2013 $M = $N.",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
discount_percent = 20
price_before_discount = 120
discount_portion = discount_percent / 100
discount_amount = price_before_discount * discount_portion
price_after_discount = price_before_discount - discount_amount
price_after_discount
{end_code}
{start_code_output}
96.0
{end_code_output}
So the new price of the lamp is \\boxed{96} dollars.
""".strip(),
    },
    {
        "question": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "reference_solution": "He is open 16*5=80 hours a week.\nSo he pays each of his employees 12.5*80=$1000 per week.\nSince there are two employees, he pays them in total 2 * $1000 = $2000 per week.\nThe utilities cost 1200*.2=$240.\nSo his total cost per week is 1200+2000+240=$3440",
        "masked_reference_solution": "He is open 16*5=M hours a week\nSo he pays each of his employees 12.5*M=$N per week\nSince there are two employees, he pays them in total 2 * $N = $O per week.\nThe utilities cost 1200*.2=$P\nSo his total cost per week is 1200+O+P=$Q",
        "generated_solution": """
The cost consists of rent, utilities, and employee salaries. Let's compute each of them separately and then add them up.
{start_code}
# rent cost
rent_per_week = 1200
# utility cost
utility_per_week = rent_per_week * 20 / 100
# employee cost
employee_work_hours = 16
work_days_per_week = 5
employee_work_hours_per_week = work_days_per_week * employee_work_hours
number_of_employees = 2
employee_cost_per_hour = 12.5
employees_cost_per_week = number_of_employees * employee_work_hours_per_week * employee_cost_per_hour
# add the three to get total cost
cost_per_week = rent_per_week + utility_per_week + employees_cost_per_week
cost_per_week
{end_code}
{start_code_output}
3440.0
{end_code_output}
Thus James's weekly expenses add up to \\boxed{3440} dollars.
""".strip(),
    },
]


# Solutions are only in text
only_text = [
    {
        "question": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
        "reference_solution": "We let x be the amount of money she put in the bank the first year.\nThen the second year was 2*x, \nand the third year was 2*2*x=4*x,\nand the fourth year was 2*2*2*x=8*x.\nAnd the total amount she put in the bank, $450, is the sum of all four years, or $450 = x+(2*x)+(4*x)+(8*x).\nThis simplifies to $450=15*x.\nAnd therefore, the amount of money she put in the bank the first year was x=$30.",
        "generated_solution": """
Let x be the amount of money Missy put in the bank the first year.
She was then doubling the amount of money every year for 4 years.
So in the second year she put 2*x amount in her bank.
In the third year she put 2*2*x = 4*x amount in her bank.
And in the fourth year she put 2*4*x = 8*x amount in her bank.
So the total amount she put in the bank is x+(2*x)+(4*x)+(8*x)=15*x.
As the problem states, this total amounts to $450.
So 450=15*x which implies x=30.
Thus, Missy put \\boxed{30} dollars in the bank the first year.
""".strip(),
    },
    {
        "question": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = (60+20) minutes = 80 minutes\nHe will spend a total of 80+10 = 90 minutes\n90 minutes is = (60/60) hours and 30 minutes = 1 hour 30 minutes = 0130 in 24-hr time\nHe has to leave 0130 hours earlier than 0900 i.e. 0900-0130 = 0730 hours",
        "generated_solution": """
Since Pete needs to take a 10 minutes walk and then a 1 hour 20 minutes train ride, he will spend a total of 1 hour and 30 minutes.
This means that he needs to leave 1 hour and 30 minutes earlier than 09:00 hours.
Subtracting 1 hour and 30 minutes from 0900 hours we get \\boxed{07:30} hours.
""".strip(),
    },
    {
        "question": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "reference_solution": "Five times the amount Mark deposited is $88 x 5 = $440.\nSo, Bryan deposited $440 - $40 = $400.",
        "generated_solution": """
Five times of what Mark deposited is $88*5=$440.
Bryan deposited $440-$40=$400.
So Bryan deposited \\boxed{400} dollars in the bank.
""".strip(),
    },
    {
        "question": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "reference_solution": "The amount of the discount is $120 x 0.2 = $24.\nThe new price will be $120 \u2013 $24 = $96.",
        "generated_solution": """
The lamp was originally priced at $120.
A 20% discount amounts to $120*20/100=$24.
So the discount reduces the price of the lamp to $120-$24=$96.
So the new price of the lamp is \\boxed{96} dollars.
""".strip(),
    },
    {
        "question": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "reference_solution": "He is open 16*5=80 hours a week.\nSo he pays each of his employees 12.5*80=$1000 per week.\nSince there are two employees, he pays them in total 2 * $1000 = $2000 per week.\nThe utilities cost 1200*.2=$240.\nSo his total cost per week is 1200+2000+240=$3440",
        "generated_solution": """
The store expense consist of rent, utilities, and employee salaries.
Let's compute each of these expenses separately at a week timescale and then add them up.
The expense due to rent is $1200.
The expense due to utilities is 20% of rent expense. Thus, it is $1200*20/100=$240.
Now we calculate the expense due to employee salaries.
Each employee works 16*5=80 hours per week.
For each employee the cost per week based on hourly wage of $12.5/hr is 12.5*80=$1000 per week.
For two employees, this amounts to 2*$1000=$2000.
Adding the cost of rent, utilities, and employee salaries amounts to $1200+$240+$2000=$3440.
Thus James's weekly expenses to run the store add up to \\boxed{3440} dollars.
""".strip(),
    },
]

# Solutions use simple Python code
only_code = [
    {
        "question": "Very early this morning, Elise left home in a cab headed for the hospital. Fortunately, the roads were clear, and the cab company only charged her a base price of $3, and $4 for every mile she traveled. If Elise paid a total of $23, how far is the hospital from her house?",
        "reference_solution": "For the distance she traveled, Elise paid 23 - 3 = 20 dollars\nSince the cost per mile is $4, the distance from Elise\u2019s house to the hospital is 20/4 = 5 miles.\nAnswer is 5",
        "masked_reference_solution": "For the distance she traveled, Elise paid 23 - 3 = M dollars\nSince the cost per mile is $4, the distance from Elise\u2019s house to the hospital is M/4 = N miles.\nAnswer is N",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
base_price = 3
per_mile_price = 4
total_paid = 23
paid_for_distance = total_paid - base_price
distance_traveled = paid_for_distance / per_mile_price
distance_traveled
{end_code}
{start_code_output}
5.0
{end_code_output}
Thus the hospital is \\boxed{5} miles from Elise's house.
""".strip(),
    },
    {
        "question": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = (60+20) minutes = 80 minutes\nHe will spend a total of 80+10 = 90 minutes\n90 minutes is = (60/60) hours and 30 minutes = 1 hour 30 minutes = 0130 in 24-hr time\nHe has to leave 0130 hours earlier than 0900 i.e. 0900-0130 = 0730 hours",
        "masked_reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = (60+20) minutes = M minutes\nHe will spend a total of M+10 = N minutes\nN minutes is = (60/60) hours and O minutes = P hour O minutes = Q in 24-hr time\nHe has to leave Q hours earlier than 0900 i.e. 0900-Q = R hours",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
from datetime import datetime, timedelta

# Time constants
walk_duration = timedelta(minutes=10)
train_duration = timedelta(hours=1, minutes=20)

# Latest arrival time in LA is 09:00 hours
arrival_time_LA = datetime.strptime("09:00", "%H:%M")

# Calculate departure time
departure_time = arrival_time_LA - walk_duration - train_duration
departure_time.strftime("%H:%M")
{end_code}
{start_code_output}
07:30
{end_code_output}
So Pete needs to leave by \\boxed{07:30} to reach LA in time.
""".strip(),
    },
    {
        "question": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "reference_solution": "Five times the amount Mark deposited is $88 x 5 = $440.\nSo, Bryan deposited $440 - $40 = $400.",
        "masked_reference_solution": "Five times the amount Mark deposited is $88 x 5 = M.\nSo, Bryan deposited M - $40 = N.",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
mark_deposit = 88
five_times_mark_deposit = 5 * mark_deposit
bryan_deposit = five_times_mark_deposit - 40
bryan_deposit
{end_code}
{start_code_output}
400
{end_code_output}
So Bryan deposited \\boxed{400} dollars in the bank.
""".strip(),
    },
    {
        "question": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "reference_solution": "The amount of the discount is $120 x 0.2 = $24.\nThe new price will be $120 \u2013 $24 = $96.",
        "masked_reference_solution": "The amount of the discount is $120 x 0.2 = M.\nThe new price will be $120 \u2013 M = N.",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
discount_percent = 20
price_before_discount = 120
discount_portion = discount_percent / 100
discount_amount = price_before_discount * discount_portion
price_after_discount = price_before_discount - discount_amount
price_after_discount
{end_code}
{start_code_output}
96.0
{end_code_output}
So the new price of the lamp is \\boxed{96} dollars.
""".strip(),
    },
    {
        "question": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "reference_solution": "He is open 16*5=80 hours a week.\nSo he pays each of his employees 12.5*80=$1000 per week.\nSince there are two employees, he pays them in total 2 * $1000 = $2000 per week.\nThe utilities cost 1200*.2=$240.\nSo his total cost per week is 1200+2000+240=$3440",
        "masked_reference_solution": "He is open 16 * 5 = M hours a week.\nSo he pays each of his employees 12.5 * M = N per week.\nSince there are two employees, he pays them in total 2 * N = O per week.\nThe utilities cost 1200 * .2 = P.\nSo his total cost per week is 1200 + O + P = Q",
        "generated_solution": """
The cost consists of rent, utilities, and employee salaries. Let's compute each of them separately and then add them up.
{start_code}
# rent cost
rent_per_week = 1200
# utility cost
utility_per_week = rent_per_week * 20 / 100
# employee cost
employee_work_hours = 16
work_days_per_week = 5
employee_work_hours_per_week = work_days_per_week * employee_work_hours
number_of_employees = 2
employee_cost_per_hour = 12.5
employees_cost_per_week = number_of_employees * employee_work_hours_per_week * employee_cost_per_hour
# add the three to get total cost
cost_per_week = rent_per_week + utility_per_week + employees_cost_per_week
cost_per_week
{end_code}
{start_code_output}
3440.0
{end_code_output}
Thus James's weekly expenses add up to \\boxed{3440} dollars.
""".strip(),
    },
]

# Solutions use sympy library
only_sympy = [
    {
        "question": 'A farmer is selling ducks and chickens at the market. He sells ducks for $10 and chickens for $8. He sells 5 chickens and some ducks. He then spends half his earnings on a new wheelbarrow. After buying it, someone sees it and agrees to pay the farmer double what the farmer paid and the farmer earns another $60. How many ducks did the farmer sell?',
        "reference_solution": "The wheelbarrow cost $30 because 60 / 2 = 30. The farmer earned $60 from selling fowl because 30 / (1/2) = 60. The farmer earned $40 selling chickens because 5 x 8 = 40. The farmer earned $20 selling ducks because 60 - 40 = 20. The farmer sold 2 ducks because 20 / 10 = 2",
        "masked_reference_solution": "The wheelbarrow cost M because 60 / 2 = M. The farmer earned $60 from selling fowl because 30 / (1/2) = 60. The farmer earned N selling chickens because 5 x 8 = N. The farmer earned O selling ducks because 60 - N = O. The farmer sold P ducks because O / 10 = P",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
import sympy as sp

# Let's denote the number of ducks sold by x
x = sp.symbols('x')

# x ducks for 10 each, 5 chickens for 8 each
total_sold = 10 * x + 5 * 8

# spends half on wheelbarrow
spent_on_wheelbarrow = total_sold / 2

# someone buys it for 60 which is double of what farmer paid
eq = sp.Eq(2 * spent_on_wheelbarrow, 60)

# solving for x we get the answer
sp.solve(eq, x)[0]
{end_code}
{start_code_output}
2
{end_code_output}
Thus the farmer sold \\boxed{2} ducks.
'''.strip(),
    },
    {
        "question": 'Yvonne brings a box of chocolates to school. Half have nuts and half do not. The students eat 80% of the ones with nuts and eat half of the ones without nuts. If there are 28 chocolates left, how many chocolates were in the box?',
        "reference_solution": "The ones with nuts that they ate was 40% of the box because 80% x .5 = 40%. The ones without nuts that they ate equaled 25% of the box because 50% x .5 = 25%. They ate 65% of the box because 40 + 25 = 65. They left 35% of the box because 100 - 65 = 35. The box had 80 chocolates because 28 / .35 = 80",
        "masked_reference_solution": "The ones with nuts that they ate was M% of the box because 80% x .5 = M%. The ones without nuts that they ate equaled N% of the box because 50% x .5 = N%. They ate O% of the box because M + N = O. They left P% of the box because 100 - O = P. The box had 80 chocolates because 28 / (P/100) = 80",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
import sympy as sp

# let's denote the amount of chocolates as x
x = sp.symbols('x')

# students ate 80% of one half and 50% of another half
students_ate = 0.8 * x / 2 + 0.5 * x / 2

# there are 28 chocolates left
eq = sp.Eq(x - students_ate, 28)

# solving for x to get the original amount of chocolates
sp.solve(eq, x)[0]
{end_code}
{start_code_output}
80.0
{end_code_output}
Thus there were \\boxed{80} chocolates in the box.
'''.strip(),
    },
    {
        "question": 'There are some jelly beans in a jar. Three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored. If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?',
        "reference_solution": "There are 750*4=3000 red jelly beans. There are 3000/3*4=4000 jelly beans in the jar.",
        "masked_reference_solution": "There are 750*4=M red jelly beans. There are M/3*4=N jelly beans in the jar.",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
from sympy import Rational, Symbol, Eq, solve

# let's denote the amount of jelly beans in the jar as x
x = Symbol('x')

# three fourths are red and one fourth of red is coconut flavored
red_beans = x * Rational(3, 4)
coconut_beans = red_beans * Rational(1, 4)

# the number of coconut jelly beans is given: 750
eq = Eq(coconut_beans, 750)

# thus the total jelly beans in the jar is
solve(eq, x)[0]
{end_code}
{start_code_output}
4000
{end_code_output}
Thus there are \\boxed{4000} jelly beans in the jar.
'''.strip(),
    },
    {
        "question": 'Abigail spent 60% of her money on food, and 25% of the remainder on her phone bill. After spending $20 on entertainment, she is left with $40. How much money did Abigail have initially?',
        "reference_solution": "Abigail had $20 + $40 = $60 left after paying her phone bill. The $60 is 100% - 25% = 75% of the remainder of her money after spending for food. Since 75% is $60, then 1% is $60/75 = $0.80. Thus, Abigail had $0.80 x 100 = $80 left after she spent money on food. The $80 is 100% - 60% = 40% of her original money. So 1% is $80/40 = $2. Therefore, Abigail initially had $2 x 100 = $200.",
        "masked_reference_solution": "Abigail had $20 + $40 = M left after paying her phone bill. The M is 100% - 25% = N% of the remainder of her money after spending for food. Since N% is $60, then 1% is $60/75 = O. Thus, Abigail had O x 100 = P left after she spent money on food. The P is 100% - 60% = Q% of her original money. So 1% is $80/40 = R. Therefore, Abigail initially had R x 100 = S.",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
import sympy as sp

# let's denote the money Abigail had initially as x
x = sp.symbols('x')

# 60% spent on food
spent_on_food = x * 0.6

# 25% of what's left spent on phone bill
spent_on_phone = (x - spent_on_food) * 0.25

# after spending another 20 she is left with 40 dollars
eq = sp.Eq(x - spent_on_food - spent_on_phone - 20, 40)

# solving for x to get the amount she had initially
sp.solve(eq, x)[0]
{end_code}
{start_code_output}
200.0
{end_code_output}
Thus, Abigail initially had \\boxed{200} dollars.
'''.strip(),
    },
    {
        'question': 'In five years, Grant will be 2/3 the age of the hospital that he is hired into. If Grant is currently 25 years old, how old is the hospital now?',
        'reference_solution': "If Grant is currently 25 years old, he will be 25+5 = 30 years old in five years. If 2/3 represents Grant's age in five years, then the fraction representing the number of years the hospital is older than Grant is 1-2/3 = 1/3. If 2/3 represents 30 years, 3/3, which is the fraction representing the age of the hospital, represents 3/3*30*3/2 = 45 years. Currently, the hospital is 45-5 = 40 years old.",
        'masked_reference_solution': "If Grant is currently 25 years old, he will be 25+5 = M years old in five years. If 2/3 represents 30 years, 3/3, which is the fraction representing the age of the hospital, represents 30*3/2 = O years. Currently, the hospital is O-5 = P years old.",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
import sympy as sp

# we need to find hospital's age now
hospital_age = sp.symbols('hospital_age')

# let's also define grant's age
grant_age = sp.symbols('grant_age')

# in five years, Grant will be 2/3 the age of the hospital
eq = sp.Eq(grant_age + 5, (hospital_age + 5) * sp.Rational(2, 3))

# we know that grant is 25 years old and need to find hospital's age
sp.solve(eq.subs(grant_age, 25), hospital_age)[0]
{end_code}
{start_code_output}
40
{end_code_output}
Thus, the hospital is \\boxed{40} years old.
'''.strip(),
    },
]

# Solutions are a mix of simple Python code (3) and sympy (2)
code_with_sympy = [
    {
        "question": "Very early this morning, Elise left home in a cab headed for the hospital. Fortunately, the roads were clear, and the cab company only charged her a base price of $3, and $4 for every mile she traveled. If Elise paid a total of $23, how far is the hospital from her house?",
        "reference_solution": "For the distance she traveled, Elise paid 23 - 3 = 20 dollars\nSince the cost per mile is $4, the distance from Elise\u2019s house to the hospital is 20/4 = 5 miles.\nAnswer is 5",
        "masked_reference_solution": "For the distance she traveled, Elise paid 23 - 3 = M dollars\nSince the cost per mile is $4, the distance from Elise\u2019s house to the hospital is M/4 = N miles.\nAnswer is N",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
base_price = 3
per_mile_price = 4
total_paid = 23
paid_for_distance = total_paid - base_price
distance_traveled = paid_for_distance / per_mile_price
distance_traveled
{end_code}
{start_code_output}
5.0
{end_code_output}
Thus the hospital is \\boxed{5} miles from Elise's house.
""".strip(),
    },
    {
        "question": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "reference_solution": "The amount of the discount is $120 x 0.2 = $24.\nThe new price will be $120 \u2013 $24 = $96.",
        "masked_reference_solution": "The amount of the discount is $120 x 0.2 = M.\nThe new price will be $120 \u2013 M = N.",
        "generated_solution": """
Let's solve this problem using Python code.
{start_code}
discount_percent = 20
price_before_discount = 120
discount_portion = discount_percent / 100
discount_amount = price_before_discount * discount_portion
price_after_discount = price_before_discount - discount_amount
price_after_discount
{end_code}
{start_code_output}
96.0
{end_code_output}
So the new price of the lamp is \\boxed{96} dollars.
""".strip(),
    },
    {
        "question": 'A farmer is selling ducks and chickens at the market. He sells ducks for $10 and chickens for $8. He sells 5 chickens and some ducks. He then spends half his earnings on a new wheelbarrow. After buying it, someone sees it and agrees to pay the farmer double what the farmer paid and the farmer earns another $60. How many ducks did the farmer sell?',
        "reference_solution": "The wheelbarrow cost $30 because 60 / 2 = 30. The farmer earned $60 from selling fowl because 30 / (1/2) = 60. The farmer earned $40 selling chickens because 5 x 8 = 40. The farmer earned $20 selling ducks because 60 - 40 = 20. The farmer sold 2 ducks because 20 / 10 = 2",
        "masked_reference_solution": "The wheelbarrow cost M because 60 / 2 = M. The farmer earned $60 from selling fowl because 30 / (1/2) = 60. The farmer earned N selling chickens because 5 x 8 = N. The farmer earned O selling ducks because 60 - N = O. The farmer sold P ducks because O / 10 = P",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
import sympy as sp

# Let's denote the number of ducks sold by x
x = sp.symbols('x')

# x ducks for 10 each, 5 chickens for 8 each
total_sold = 10 * x + 5 * 8

# spends half on wheelbarrow
spent_on_wheelbarrow = total_sold / 2

# someone buys it for 60 which is double of what farmer paid
eq = sp.Eq(2 * spent_on_wheelbarrow, 60)

# solving for x we get the answer
sp.solve(eq, x)[0]
{end_code}
{start_code_output}
2
{end_code_output}
Thus the farmer sold \\boxed{2} ducks.
'''.strip(),
    },
    {
        "question": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "reference_solution": "He is open 16*5=80 hours a week.\nSo he pays each of his employees 12.5*80=$1000 per week.\nSince there are two employees, he pays them in total 2 * $1000 = $2000 per week.\nThe utilities cost 1200*.2=$240.\nSo his total cost per week is 1200+2000+240=$3440",
        "masked_reference_solution": "He is open 16 * 5 = M hours a week.\nSo he pays each of his employees 12.5 * M = N per week.\nSince there are two employees, he pays them in total 2 * N = O per week.\nThe utilities cost 1200 * .2 = P.\nSo his total cost per week is 1200 + O + P = Q",
        "generated_solution": """
The cost consists of rent, utilities, and employee salaries. Let's compute each of them separately and then add them up.
Let's solve this problem using Python code.
{start_code}
# rent cost
rent_per_week = 1200
# utility cost
utility_per_week = rent_per_week * 20 / 100
# employee cost
employee_work_hours = 16
work_days_per_week = 5
employee_work_hours_per_week = work_days_per_week * employee_work_hours
number_of_employees = 2
employee_cost_per_hour = 12.5
employees_cost_per_week = number_of_employees * employee_work_hours_per_week * employee_cost_per_hour
# add the three to get total cost
cost_per_week = rent_per_week + utility_per_week + employees_cost_per_week
cost_per_week
{end_code}
{start_code_output}
3440.0
{end_code_output}
Thus James's weekly expenses add up to \\boxed{3440} dollars.
""".strip(),
    },
    {
        "question": 'Yvonne brings a box of chocolates to school. Half have nuts and half do not. The students eat 80% of the ones with nuts and eat half of the ones without nuts. If there are 28 chocolates left, how many chocolates were in the box?',
        "reference_solution": "The ones with nuts that they ate was 40% of the box because 80% x .5 = 40%. The ones without nuts that they ate equaled 25% of the box because 50% x .5 = 25%. They ate 65% of the box because 40 + 25 = 65. They left 35% of the box because 100 - 65 = 35. The box had 80 chocolates because 28 / .35 = 80",
        "masked_reference_solution": "The ones with nuts that they ate was M% of the box because 80% x .5 = M%. The ones without nuts that they ate equaled N% of the box because 50% x .5 = N%. They ate O% of the box because M + N = O. They left P% of the box because 100 - O = P. The box had 80 chocolates because 28 / (P/100) = 80",
        'generated_solution': '''
Let's solve this problem using Python's sympy library.
{start_code}
import sympy as sp

# let's denote the amount of chocolates as x
x = sp.symbols('x')

# students ate 80% of one half and 50% of another half
students_ate = 0.8 * x / 2 + 0.5 * x / 2

# there are 28 chocolates left
eq = sp.Eq(x - students_ate, 28)

# solving for x to get the original amount of chocolates
sp.solve(eq, x)[0]
{end_code}
{start_code_output}
80.0
{end_code_output}
Thus there were \\boxed{80} chocolates in the box.
'''.strip(),
    },
]


generate_masked = [
    {
        "question": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
        "reference_solution": "We let x be the amount of money she put in the bank the first year.\nThen the second year was 2*x, \nand the third year was 2*2*x=4*x,\nand the fourth year was 2*2*2*x=8*x.\nAnd the total amount she put in the bank, $450, is the sum of all four years, or $450 = x+(2*x)+(4*x)+(8*x).\nThis simplifies to $450=15*x.\nAnd therefore, the amount of money she put in the bank the first year was x=$30.",
        "generated_solution": """
We let x be the amount of money she put in the bank the first year.\nThen the second year was 2*x, \nand the third year was 2*2*x=M*x,\nand the fourth year was 2*2*2*x=N*x.\nAnd the total amount she put in the bank, $450, is the sum of all four years, or $450 = x+(2*x)+(M*x)+(N*x).\nThis simplifies to $450=(1 + 2 + M + N)*x.\nAnd therefore, the amount of money she put in the bank the first year was x=P dollars.
""".strip(),
    },
    {
        "question": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = (60+20) minutes = 80 minutes\nHe will spend a total of 80+10 = 90 minutes\n90 minutes is = (60/60) hours and 30 minutes = 1 hour 30 minutes = 0130 in 24-hr time\nHe has to leave 0130 hours earlier than 0900 i.e. 0900-0130 = 0730 hours",
        "generated_solution": """
There are 60 minutes in an hour so 1 hour 20 minutes = M minutes\nHe will spend a total of M + 10 = N minutes\nN minutes is = O hours and P minutes = O hour P minutes = Q in 24-hr time\nHe has to leave Q hours earlier than 0900 i.e. 0900 - Q = R hours""".strip(),
    },
    {
        "question": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "reference_solution": "Five times the amount Mark deposited is $88 x 5 = $440.\nSo, Bryan deposited $440 - $40 = $400.",
        "generated_solution": """
Five times the amount Mark deposited is $88 x 5 = M.\nSo, Bryan deposited M - $40 = N.""".strip(),
    },
    {
        "question": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "reference_solution": "The amount of the discount is $120 x 0.2 = $24.\nThe new price will be $120 \u2013 $24 = $96.",
        "generated_solution": """
The amount of the discount is $120 x 0.2 = M.\nThe new price will be $120 \u2013 M = N.
""".strip(),
    },
    {
        "question": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "reference_solution": "He is open 16*5=80 hours a week.\nSo he pays each of his employees 12.5*80=$1000 per week.\nSince there are two employees, he pays them in total 2 * $1000 = $2000 per week.\nThe utilities cost 1200*.2=$240.\nSo his total cost per week is 1200+2000+240=$3440",
        "generated_solution": """
He is open 16 * 5 = M hours a week.\nSo he pays each of his employees 12.5 * M = N per week.\nSince there are two employees, he pays them in total 2 * N = O per week.\nThe utilities cost 1200 * .2 = P.\nSo his total cost per week is 1200 + O + P = Q
""".strip(),
    },
]


examples_map = {
    "gsm8k_text_with_code": text_with_code,
    "gsm8k_generate_masked": generate_masked,
    "gsm8k_only_text": only_text,
    "gsm8k_only_code": only_code,
    "gsm8k_only_sympy": only_sympy,
    "gsm8k_code_with_sympy": code_with_sympy,
}
