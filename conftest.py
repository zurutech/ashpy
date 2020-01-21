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

"""pytest configuration."""
import os
import shutil

import pytest
import tensorflow  # pylint: disable=import-error

import ashpy


@pytest.fixture(autouse=True)
def add_common_namespaces(doctest_namespace):
    """Add the common namespace to all tests."""
    doctest_namespace["tf"] = tensorflow
    doctest_namespace["trainers"] = ashpy.trainers
    doctest_namespace["models"] = ashpy.models
    doctest_namespace["metrics"] = ashpy.metrics
    doctest_namespace["layers"] = ashpy.layers
    doctest_namespace["losses"] = ashpy.losses
    doctest_namespace["callbacks"] = ashpy.callbacks


@pytest.fixture(scope="function")
def save_dir():
    """Add the save_dir parameter to tests."""
    m_save_dir = "testlog/savedir"

    # Clean before
    if os.path.exists(m_save_dir):
        shutil.rmtree(m_save_dir)
        assert not os.path.exists(m_save_dir)

    yield m_save_dir

    # teardown
    if os.path.exists(m_save_dir):
        shutil.rmtree(m_save_dir)
        assert not os.path.exists(m_save_dir)
