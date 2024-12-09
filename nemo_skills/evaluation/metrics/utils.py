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
# See the License for the specific lang

import json


def read_predictions(predictions):
    data = []
    for prediction in predictions:
        prediction_dict = json.loads(prediction)
        data.append(prediction_dict)

    return data


def is_correct_judgement(judgement):
    if 'Judgement:' not in judgement:
        return False  # improper judgement format, so have to judge as false
    verdict = judgement.split('Judgement:')[-1].strip()
    return verdict.lower() == 'yes'
