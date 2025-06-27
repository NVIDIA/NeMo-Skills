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


# Derived from https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py


import json

def _get_language_specific_hint(test_category):
    if test_category == "java":
        return " Note that the provided function is in Java 8 SDK syntax."
    elif test_category == "javascript":
        return " Note that the provided function is in JavaScript syntax."
    else:
        return " Note that the provided function is in Python 3 syntax."


def func_doc_language_specific_pre_processing(function, test_category):
    if len(function) == 0:
        return function

    assert type(function) == list
    for item in function:
        # Add language specific hints to the function description
        item["description"] = item["description"] + _get_language_specific_hint(
            test_category
        )
        # Process the parameters
        properties = item["parameters"]["properties"]
        if test_category == "java":
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key][
                        "description"
                    ] += " This parameter can be of any type of Java object in string representation."
                else:
                    value[
                        "description"
                    ] += f" This is Java {value['type']} type parameter in string representation."
                if value["type"] == "ArrayList" or value["type"] == "Array":
                    value[
                        "description"
                    ] += f" The list elements are of type {value['items']['type']}; they are not in string representation."
                    del value["items"]

                value["type"] = "string"

        elif test_category == "javascript":
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key][
                        "description"
                    ] += " This parameter can be of any type of JavaScript object in string representation."
                else:
                    value[
                        "description"
                    ] += f" This is JavaScript {value['type']} type parameter in string representation."
                if value["type"] == "array":
                    value[
                        "description"
                    ] += f" The list elements are of type {value['items']['type']}; they are not in string representation."
                    del value["items"]

                if value["type"] == "dict":
                    if "properties" in value:  # not every dict has properties
                        value[
                            "description"
                        ] += f" The dictionary entries have the following schema; they are not in string representation. {json.dumps(value['properties'])}"
                        del value["properties"]

                value["type"] = "string"

    return function