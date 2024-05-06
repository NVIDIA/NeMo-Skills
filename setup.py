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

from setuptools import find_packages, setup

setup(
    name="nemo_skills",
    version="0.2.0",
    description="NeMo Skills - a project to improve skills of LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    url="https://github.com/Kipok/NeMo-Skills",
    packages=find_packages(include=["nemo_skills*"]),
    python_requires=">=3.8",
    install_requires=[
        'hydra-core',
        'tqdm',
        'pyyaml',
        'numpy',
        'requests',
        'backoff',
        'sympy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
