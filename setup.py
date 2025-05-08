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
import importlib.util
import os
from itertools import chain

spec = importlib.util.spec_from_file_location('version', 'nemo_skills/version.py')
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__version__ = package_info.__version__


def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


# Read the requirements from the requirements.txt file
requirements = parse_requirements('requirements/main.txt')

extras_require = {
    'core': requirements,
}
install_requires = list(chain(*[extras_require['core'],]))
extras_require['all'] = list(chain(*list(extras_require.values())))

setup(
    name='nemo_skills',
    version=__version__,
    description="NeMo Skills - a project to improve skills of LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    url="https://github.com/NVIDIA/NeMo-Skills",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    include_package_data=True,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'ns=nemo_skills.pipeline.cli:app',
        ],
    },
)
