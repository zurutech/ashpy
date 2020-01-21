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

"""
Test Metrics with the various trainers.

TODO: Adversarial Encoder
"""

import json
import operator
import pathlib

import pytest
from ashpy.metrics import (
    ClassifierLoss,
    InceptionScore,
    SlicedWassersteinDistance,
    SSIM_Multiscale,
)
from ashpy.models.gans import ConvDiscriminator

from tests.utils.fake_training_loop import (
    fake_adversarial_training_loop,
    fake_classifier_taining_loop,
)

DEFAULT_LOGDIR = "log"
TEST_MATRIX = {
    "adversarial_trainer": [
        fake_adversarial_training_loop,
        {
            "image_resolution": [256, 256],
            "layer_spec_input_res": (8, 8),
            "layer_spec_target_res": (8, 8),
            "channels": 3,
        },
        [
            SlicedWassersteinDistance(resolution=256),
            SSIM_Multiscale(),
            InceptionScore(
                # Fake inception model
                ConvDiscriminator(
                    layer_spec_input_res=(299, 299),
                    layer_spec_target_res=(7, 7),
                    kernel_size=(5, 5),
                    initial_filters=16,
                    filters_cap=32,
                    output_shape=10,
                )
            ),
        ],
    ],
    "classifier_trainer": [
        fake_classifier_taining_loop,
        {},
        [ClassifierLoss(model_selection_operator=operator.lt)],
    ],
}

TEST_PARAMS = [TEST_MATRIX[k] for k in TEST_MATRIX]
TEST_IDS = [k for k in TEST_MATRIX]


@pytest.fixture(scope="module")
def cleanup():
    """Remove the default logdir before and after testing."""
    default_log_dir = pathlib.Path(DEFAULT_LOGDIR)
    if default_log_dir.exists():
        shutil.rmtree(default_log_dir)
    yield "Cleanup"
    if default_log_dir.exists():
        shutil.rmtree(default_log_dir)


@pytest.mark.parametrize(
    ["training_loop", "loop_args", "metrics"], TEST_PARAMS, ids=TEST_IDS
)
def test_metrics(training_loop, loop_args, metrics, tmpdir, cleanup):
    """
    Test the integration between metrics and trainer.

    GIVEN a correctly instantiated trainer
    GIVEN some training has been done
        THEN there should not be any logs inside the default log folder
        THEN there exists a logdir folder for each metric
        THEN there inside in each folder there's the JSON file w/ the metric logs
        THEN in the file there are the correct keys

    """
    training_completed = training_loop(logdir=tmpdir, metrics=metrics, **loop_args)

    assert training_completed
    assert not pathlib.Path(DEFAULT_LOGDIR).exists()  # Assert absence of side effects
    # Assert there exists folder for each metric
    for metric in metrics:
        metric_dir = pathlib.Path(tmpdir).joinpath("best", metric.name)
        assert metric_dir.exists()
        json_path = metric_dir.joinpath(f"{metric.name}.json")
        assert json_path.exists()
        with open(json_path, "r") as fp:
            metric_data = json.load(fp)

            # Assert the metric data contains the expected keys
            assert metric.name in metric_data
            assert "step" in metric_data
