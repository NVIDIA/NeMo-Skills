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
        "question": """Read the table regarding "Coin collections" and then write Python code to answer the following question:

Question: Some friends discussed the sizes of their coin collections. What is the mean of the numbers?""",
        "table": """
Name | Number of coins
Braden | 76
Camilla | 94
Rick | 86
Mary | 84
Hector | 80
Devin | 83
Emily | 82
Avery | 87
""".strip(),
        "reference_solution": "Read the numbers from the table.\n\n76, 94, 86, 84, 80, 83, 82, 87\n\nFirst, count how many numbers are in the group.\n\nThere are 8 numbers.\n\nNow add all the numbers together:\n\n76 + 94 + 86 + 84 + 80 + 83 + 82 + 87 = 672\n\nNow divide the sum by the number of numbers:\n\n672 \u00f7 8 = 84\n\nThe mean is 84.",
        "generated_solution": """
Let's solve this problem using Python code and the values from the table.
{start_code}
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
result = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)
result
{end_code}
{start_code_output}
84.0
{end_code_output}
So the mean of the sizes of their coin collections is \\boxed{84} coins.
""".strip(),
    },
    {
        "question": """Read the table regarding "" and then write Python code to answer the following question:

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]""",
        "table": """
Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200
""".strip(),
        "reference_solution": "At the price of $155, the quantity demanded is greater than the quantity supplied. There is not enough of the good or service for sale at that price. So, there is a shortage.\n",
        "generated_solution": """
Let's solve this problem using Python code and the values from the table.
{start_code}
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_price_155 > quantity_supplied_price_155:
    result = 'shortage'
else:
    result = 'surplus'
result
{end_code}
{start_code_output}
shortage
{end_code_output}
So there is a \\boxed{shortage} at a price of 155 dollars.
""".strip(),
    },
    {
        "question": """Read the table regarding "Cans of food collected" and then write Python code to answer the following question:

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?""",
        "table": """
Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9
""".strip(),
        "reference_solution": "Read the numbers from the table.\n\n7, 4, 7, 8, 9\n\nFirst, arrange the numbers from least to greatest:\n\n4, 7, 7, 8, 9\n\nNow find the number in the middle.\n\n4, 7, 7, 8, 9\n\nThe number in the middle is 7.\n\nThe median is 7.",
        "generated_solution": """
Let's solve this problem using Python code and the values from the table.
{start_code}
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
result = (cans[middle1] + cans[middle2]) / 2
result
{end_code}
{start_code_output}
7.0
{end_code_output}
The median number of cans collected is \\boxed{7}.
""".strip(),
    },
    {
        "question": """Read the table regarding "" and then write Python code to answer the following question:

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']""",
        "table": """
toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00
""".strip(),
        "reference_solution": "Add the price of a toy guitar and the price of a set of juggling balls:\n\n$8.23 + $5.01 = $13.24\n\n$13.24 is less than $13.50. Lorenzo does have enough money.",
        "generated_solution": """
Let's solve this problem using Python code and the values from the table.
{start_code}
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls + guitar_price:
    result = 'yes'
else:
    result = 'no'
result
{end_code}
{start_code_output}
yes
{end_code_output}
\\boxed{Yes}, Lorenzo does have enough money to buy a toy guitar and juggling balls.
""".strip(),
    },
]

examples_map = {
    "tabmwp_text_with_code": text_with_code,
}
