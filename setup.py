# Copyright 2019 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package AshPy."""
import re

from setuptools import find_packages, setup

# Meta
INIT_PY = open("src/ashpy/__init__.py").read()
METADATA = dict(re.findall(r"__([a-z]+)__ = \"([^\"]+)\"", INIT_PY))

# Info
README = open("README.md").read()

# Requirements
INSTALL_REQUIREMENTS = ["tensorflow>=2.1.0", "tensorflow_hub"]

setup(
    author_email=METADATA["email"],
    author=METADATA["author"],
    classifiers=["Programming Language :: Python :: 3.7"],
    description=(
        "TensorFlow 2.0 library for distributed training, "
        "evaluation, model selection, and fast prototyping."
    ),
    include_package_data=True,
    install_requires=INSTALL_REQUIREMENTS,
    keywords=["ashpy", "ai", "tensorflow", "tensorflow-2.0", "deeplearning"],
    license="Apache License, Version 2.0",
    long_description_content_type="text/markdown",
    long_description=README,
    name="ashpy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    url=METADATA["url"],
    version=METADATA["version"],
    zip_safe=False,
)
