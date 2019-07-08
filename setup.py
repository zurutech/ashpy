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

from setuptools import find_packages, setup

# Info
REPO = "https://github.com/zurutech/ashpy"
README = open("README.md").read()
DOCLINK = f"""
Documentation
-------------

The full documentation is at {REPO}.
"""

# Requirements
REQUIREMENTS = ["numpy>=1.16.3", "tensorflow_hub>=0.4.0"]
TEST_REQUIREMENTS = ["pytest"]

setup(
    name="ashpy",
    version="1.0.0",
    description="TensorFlow 2.0 library for distributed training, evaluation, model selection, and fast prototyping.",
    long_description=README + "\n\n" + DOCLINK,
    long_description_content_type="text/markdown",
    author=["Machine Learning Team @ Zuru Tech"],
    author_email="ml@zuru.tech",
    url=REPO,
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
    license="Apache License, Version 2.0",
    zip_safe=False,
    keywords=["ashpy", "ai", "tensorflow", "tensorflow-2.0", "deeplearning"],
    classifiers=["Programming Language :: Python :: 3.7"],
)
