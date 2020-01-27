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

import operator
import os
import shutil

import pytest
import tensorflow  # pylint: disable=import-error

import ashpy
from ashpy.metrics import (
    ClassifierLoss,
    InceptionScore,
    SlicedWassersteinDistance,
    SSIM_Multiscale,
)
from tests.utils.fake_training_loop import (
    fake_adversarial_training_loop,
    fake_classifier_training_loop,
)


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


# ------------------------------------------------------------------------------------

TEST_MATRIX = {
    # NOTE: Always pass metrics as Tuple, Trainers produce side effects!
    "adversarial_trainer": [
        fake_adversarial_training_loop,
        {
            "image_resolution": [256, 256],
            "layer_spec_input_res": (8, 8),
            "layer_spec_target_res": (8, 8),
            "channels": 3,
            "output_shape": 1,
            "measure_performance_freq": 1,
        },
        (
            SlicedWassersteinDistance(resolution=256),
            SSIM_Multiscale(),
            InceptionScore(
                # Fake inception model
                ashpy.models.gans.ConvDiscriminator(
                    layer_spec_input_res=(299, 299),
                    layer_spec_target_res=(7, 7),
                    kernel_size=(5, 5),
                    initial_filters=16,
                    filters_cap=32,
                    output_shape=10,
                )
            ),
        ),
    ],
    "classifier_trainer": [
        fake_classifier_training_loop,
        {"measure_performance_freq": 1},
        (ClassifierLoss(model_selection_operator=operator.lt),),
    ],
}

TRAINING_IDS = [k for k in TEST_MATRIX]
LOOPS = [TEST_MATRIX[k] for k in TEST_MATRIX]


@pytest.fixture(scope="function", params=LOOPS, ids=TRAINING_IDS)
def fake_training(request):
    """Fixture used to generate fake training for the tests."""
    training_loop, loop_args, metrics = request.param
    assert len(metrics) in [1, 3]
    return (training_loop, loop_args, list(metrics))
