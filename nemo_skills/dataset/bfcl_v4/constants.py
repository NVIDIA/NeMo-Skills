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

from pathlib import Path


ALL_AVAILABLE_MEMORY_BACKENDS = [
    "kv",
    "vector",
    "rec_sum",
]

NON_LIVE_CATEGORY = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    # "exec_simple",
    # "exec_parallel",
    # "exec_multiple",
    # "exec_parallel_multiple",
    # "rest",
    # "sql",
    # "chatable",
]
LIVE_CATEGORY = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]
MULTI_TURN_CATEGORY = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    # "multi_turn_composite",
]
WEB_SEARCH_CATEGORY = [
    "web_search_base",
    "web_search_no_snippet",
]

MEMORY_CATEGORY = [f"memory_{backend}" for backend in ALL_AVAILABLE_MEMORY_BACKENDS]


SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY
NON_SCORING_CATEGORY = ["format_sensitivity"]

ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + AGENTIC_CATEGORY
ALL_CATEGORIES = ALL_SCORING_CATEGORIES + NON_SCORING_CATEGORY

TEST_COLLECTION_MAPPING = {
    "all": ALL_CATEGORIES,
    "all_scoring": ALL_SCORING_CATEGORIES,
    "multi_turn": MULTI_TURN_CATEGORY,
    "single_turn": SINGLE_TURN_CATEGORY,
    "live": LIVE_CATEGORY,
    "non_live": NON_LIVE_CATEGORY,
    "non_python": [
        "simple_java",
        "simple_javascript",
    ],
    "python": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "memory": MEMORY_CATEGORY,
    "web_search": WEB_SEARCH_CATEGORY,
    "agentic": AGENTIC_CATEGORY,
}


MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


# Repo relative paths

MULTI_TURN_FUNC_DOC_PATH = Path("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc")
DATA_FOLDER_PATH = Path("berkeley-function-call-leaderboard/bfcl_eval/data")
