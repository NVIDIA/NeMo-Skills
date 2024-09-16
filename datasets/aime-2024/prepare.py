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


import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def get_all_problem_links():
    # URLs of the webpage
    urls = [
        "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems",
        "https://artofproblemsolving.com/wiki/index.php/2024_AIME_II",
    ]
    problem_links = set()

    # Get the webpage content
    for url in urls:
        response = requests.get(url)
        html_content = response.content

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all links that contain 'Problem'
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if 'Problem_' in href and ('2024_AIME_I_Problems' in href or '2024_AIME_II_Problems' in href):
                problem_links.add("https://artofproblemsolving.com" + href)

    def extract_problem_number(url):
        match = re.search(r'Problem_(\d+)$', url)
        return int(match.group(1)) if match else float('inf')

    def extract_problem_set(url):
        match = re.search(r'(2024_AIME_[I|II]_Problems)', url)
        return match.group(1) if match else ''

    urls = list(problem_links)
    #### we sort the urls links for 2024 I and 2024 II with problem order
    sorted_urls = sorted(urls, key=lambda url: (extract_problem_set(url), extract_problem_number(url)))

    return sorted_urls


def get_question_or_solution(url, choice):
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content
    def replace_img_with_latex(soup):
        for img in soup.find_all('img', class_='latex'):
            latex_code = img.get('alt')
            img.replace_with(f" {latex_code} ")
        return soup

    def find_exact_answer(solution):
        # Define the regex pattern for \boxed{content}
        pattern = re.compile(r'\\boxed{.*?}|\\framebox{.*?}')

        # Find all matches of the pattern in the string
        matches = re.findall(pattern, solution)

        # Find the final boxed information, which should be the answer
        extracted_digits = re.findall(r'\d+', matches[-1])
        answer = ''.join(extracted_digits)
        return answer

    soup = BeautifulSoup(html_content, 'html.parser')
    soup = replace_img_with_latex(soup)

    if choice == 'solution':
        # Extract the headline for solution
        pattern = re.compile(r'^Solution_1')
        ### deal with one edge case
        if "2024_AIME_II_Problems/Problem_10" in url:
            pattern = re.compile(r'^Solution_2')
        problem_header = (
            soup.find('span', id='Solution 1') or soup.find('span', id='Solution') or soup.find('span', id=pattern)
        )
        parent_h2 = problem_header.find_parent('h2')

    elif choice == 'question':
        ### Extract the headline for problem
        problem_header = soup.find('span', id='Problem')
        parent_h2 = problem_header.find_parent('h2')

    # Find the next sibling elements which should be paragraphs
    next_sibling = parent_h2.find_next_sibling()
    elements = []
    while next_sibling:

        if next_sibling.name == 'p':
            # Extract text including LaTeX from img alt attributes
            text = ''
            for content in next_sibling.contents:
                if content.name == 'img' and 'alt' in content.attrs:
                    text += ' ' + content['alt'] + ' '
                else:
                    text += str(content)
            elements.append(text.strip())
        elif next_sibling.name == 'dl':
            dt_elements = next_sibling.find_all('dt')
            for dt in dt_elements:
                elements.append(dt.get_text())
        elif next_sibling.name == 'ul':
            dt_elements = next_sibling.find_all('li')
            for dt in dt_elements:
                elements.append(dt.get_text())
        next_sibling = next_sibling.find_next_sibling()
        ### we only need to find one solution
        if next_sibling.name == 'h2':
            break

    if choice == 'solution':
        solution = ' '.join(elements)
        exact_answer = find_exact_answer(solution)
        return solution, exact_answer
    elif choice == 'question':
        filtered_elements = [ele for ele in elements if not ele.startswith('[asy]')]
        question = ' '.join(filtered_elements)
        return question


if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent
    original_file = str(data_folder / "original_test.json")
    data_folder.mkdir(exist_ok=True)
    output_file = str(data_folder / "test.jsonl")

    data = []

    links = get_all_problem_links()
    for url in links:
        question = get_question_or_solution(url, choice='question')
        solution, expected_answer = get_question_or_solution(url, choice='solution')
        expected_answer = expected_answer.lstrip('0')
        new_entry = {}

        if url.endswith("2024_AIME_I_Problems/Problem_12"):
            expected_answer = '385'

        new_entry["question"] = question
        new_entry["expected_answer"] = expected_answer
        new_entry["reference_solution"] = solution

        data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
