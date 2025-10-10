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

from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://0.0.0.0:5000/v1", timeout=None)
api_model = client.models.list().data[0].id

print("API model: ", api_model)
print("client.models.list(): ", client.models.list())

response = client.chat.completions.create(
    model=api_model,
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.0,
    max_tokens=4,
    top_p=1.0,
    n=1,
    stream=False,
)

print("Response: ", response)

print(response.choices[0].message.content)
