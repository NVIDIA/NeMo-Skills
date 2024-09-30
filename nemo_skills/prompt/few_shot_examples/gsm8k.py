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


# Source https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
# Derived from the CoT paper - https://arxiv.org/pdf/2201.11903 - Table 20
gsm8k_standard_few_shot = [
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is \\boxed{6}.",
    },
    {
        "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is \\boxed{5}.",
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is \\boxed{39}.",
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is \\boxed{8}.",
    },
    {
        "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "solution": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is \\boxed{9}.",
    },
    {
        "problem": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "solution": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is \\boxed{29}.",
    },
    {
        "problem": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "solution": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is \\boxed{33}.",
    },
    {
        "problem": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "solution": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is \\boxed{8}.",
    },
]


gsm8k_text_with_code = [
    {
        "problem": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
        "solution": """
Missy started with some amount of money in the first year.
She was then doubling the amount of money every year for 4 years and ended up with $450.
Let's write down an equation for this problem and solve it using sympy.
{code_begin}
from sympy import solve, symbols
first_year_money = symbols('first_year_money')
second_year_money = 2 * first_year_money
third_year_money = 2 * second_year_money
fourth_year_money = 2 * third_year_money
# Solve for first year money given that the combined saving is 450
result = solve(first_year_money + second_year_money + third_year_money + fourth_year_money - 450, first_year_money)[0]
result
{code_end}
{code_output_begin}

completed
[stdout]
30
[/stdout]{code_output_end}

Thus Missy put \\boxed{30} dollars in the bank the first year.
""".strip(),
    },
    {
        "problem": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "solution": """
Since Pete needs to take a 10 minutes walk and then a 1 hour 20 minutes train ride, he will spend a total of 1 hour and 30 minutes.
This means that he needs to leave 1 hour and 30 minutes earlier than 09:00 hours.
Subtracting 1 hour and 30 minutes from 09:00 hours we get \\boxed{07:30} hours.
""".strip(),
    },
    {
        "problem": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "solution": """
Let's solve this problem using Python code.
{code_begin}
mark_deposit = 88
five_times_mark_deposit = 5 * mark_deposit
bryan_deposit = five_times_mark_deposit - 40
bryan_deposit
{code_end}{code_output_begin}

completed
[stdout]
400
[/stdout]{code_output_end}

So Bryan deposited \\boxed{400} dollars in the bank.
""".strip(),
    },
    {
        "problem": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "solution": """
Let's solve this problem using Python code.
{code_begin}
discount_percent = 20
price_before_discount = 120
discount_portion = discount_percent / 100
discount_amount = price_before_discount * discount_portion
price_after_discount = price_before_discount - discount_amount
price_after_discount
{code_end}{code_output_begin}

completed
[stdout]
96.0
[/stdout]{code_output_end}

So the new price of the lamp is \\boxed{96} dollars.
""".strip(),
    },
    {
        "problem": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "solution": """
The cost consists of rent, utilities, and employee salaries. Let's compute each of them separately and then add them up.
{code_begin}
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
{code_end}{code_output_begin}

completed
[stdout]
3440.0
[/stdout]{code_output_end}

Thus James's weekly expenses add up to \\boxed{3440} dollars.
""".strip(),
    },
]


# Solutions are only in text
gsm8k_text_detailed = [
    {
        "problem": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
        "solution": """
Let $x$ be the amount of money Missy put in the bank the first year.
She was then doubling the amount of money every year for 4 years.
So in the second year she put $2*x$ amount in her bank.
In the third year she put $2*2*x = 4*x$ amount in her bank.
And in the fourth year she put $2*4*x = 8*x$ amount in her bank.
So the total amount she put in the bank is $x + (2*x) + (4*x) + (8*x) = 15*x$.
As the problem states, this total amounts to $450.
So $450 = 15*x$ which implies $x = 30$.
Thus, Missy put \\boxed{30} dollars in the bank the first year.
""".strip(),
    },
    {
        "problem": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "solution": """
Since Pete needs to take a 10 minutes walk and then a 1 hour 20 minutes train ride, he will spend a total of 1 hour and 30 minutes.
This means that he needs to leave 1 hour and 30 minutes earlier than 09:00 hours.
Subtracting 1 hour and 30 minutes from 0900 hours we get \\boxed{07:30} hours.
""".strip(),
    },
    {
        "problem": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "solution": """
Five times of what Mark deposited is $88 * 5 = 440$.
Bryan deposited $440 - 40 = 400$.
So Bryan deposited \\boxed{400} dollars in the bank.
""".strip(),
    },
    {
        "problem": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "solution": """
The lamp was originally priced at $120.
A 20% discount amounts to $120 * 20 / 100 = 24$.
So the discount reduces the price of the lamp to $120 - 24 = 96$.
So the new price of the lamp is \\boxed{96} dollars.
""".strip(),
    },
    {
        "problem": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
        "solution": """
The store expense consist of rent, utilities, and employee salaries.
Let's compute each of these expenses separately at a week timescale and then add them up.
The expense due to rent is $1200.
The expense due to utilities is 20% of rent expense. Thus, it is $1200 * 20 / 100 = 240$.
Now we calculate the expense due to employee salaries.
Each employee works 16*5=80 hours per week.
For each employee the cost per week based on hourly wage of $12.5/hr is $12.5 * 80 = 1000$ per week.
For two employees, this amounts to $2 * 1000 = 2000$.
Adding the cost of rent, utilities, and employee salaries amounts to $1200 + 240 + 2000 = 3440$.
Thus James's weekly expenses to run the store add up to \\boxed{3440} dollars.
""".strip(),
    },
]

gsm8k_problem_augmentation = [
    {
        'problem': 'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
        'augmented_problem': 'Aiden has $35. He purchased eight pencils for $2 each and a notebook for $5. How much money does he have remaining?',
    },
    {
        'problem': 'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
        'augmented_problem': 'Sarah collected 72 seashells during her beach vacation. On Thursday, she gave 15 seashells to her friend as a souvenir. On Friday, she found 8 more seashells while exploring the shore. How many seashells did Sarah have at the end of Friday?',
    },
    {
        'problem': 'Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?',
        'augmented_problem': 'Samantha and David are preparing for their upcoming science fair project. They have four different experiments to conduct and a research paper to write. Each experiment is estimated to take 2 hours, and the research paper will require 8 hours to complete. To stay focused and productive, they plan to take a 15-minute break for every 1.5 hours of work and have three 20-minute snack breaks each day. Additionally, they allocate 45 minutes for lunch each day. If they want to limit their daily study time to 5 hours, how many days should they plan to work on their project over the next two weeks?',
    },
    {
        'problem': 'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
        'augmented_problem': 'Tom has 50 marbles, and his friend Jerry has 65 marbles. If they decide to play a game and bet 20 marbles each, how many marbles will they have left in total after the game?',
    },
    {
        'problem': 'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
        'augmented_problem': 'In a garden, there were 12 flowers. Every morning for a week (from Monday to Sunday), 3 more flowers were planted. How many flowers are there in the garden now?',
    },
    {
        'problem': 'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
        'augmented_problem': 'Sarah had 35 marbles. She gave some marbles to her friend Emma. Now Sarah has 18 marbles left. How many marbles did Sarah give to Emma?',
    },
    {
        'problem': 'Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately at the rate of three pens for $2. How much profit did he make in total, in dollars?',
        'augmented_problem': 'Amy purchased 8 crates, each containing 24 colorful markers, for $12 per crate. She decided to create sets of 4 markers each and sell them for $2 per set. The remaining markers she sold individually at a rate of 5 markers for $3. Calculate the total profit Amy made, in dollars.',
    },
    {
        'problem': 'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
        'augmented_problem': 'In a garden, there are 25 rose bushes. The gardener plans to plant some more rose bushes today. After planting, there will be a total of 40 rose bushes in the garden. How many rose bushes will the gardener plant today?',
    },
]


examples_map = {
    "gsm8k_standard_few_shot": gsm8k_standard_few_shot,
    "gsm8k_text_with_code": gsm8k_text_with_code,
    "gsm8k_problem_augmentation": gsm8k_problem_augmentation,
}
