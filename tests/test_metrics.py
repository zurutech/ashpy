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
import shutil
from typing import List

import pytest
import tensorflow as tf
from ashpy.metrics import (
    ClassifierLoss,
    ClassifierMetric,
    InceptionScore,
    Metric,
    SlicedWassersteinDistance,
    SSIM_Multiscale,
)
from ashpy.models.gans import ConvDiscriminator

from tests.utils.fake_training_loop import (
    fake_adversarial_training_loop,
    fake_classifier_taining_loop,
)

DEFAULT_LOGDIR = "log"
TEST_MATRIX_METRICS_LOG = {
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

TEST_PARAMS_METRICS_LOG = [TEST_MATRIX_METRICS_LOG[k] for k in TEST_MATRIX_METRICS_LOG]
TEST_IDS_METRICS_LOG = [k for k in TEST_MATRIX_METRICS_LOG]


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
    ["training_loop", "loop_args", "metrics"],
    TEST_PARAMS_METRICS_LOG,
    ids=TEST_IDS_METRICS_LOG,
)
def test_metrics_log(training_loop, loop_args, metrics, tmpdir, cleanup):
    """
    Test that trainers correctly create metrics log files.

    GIVEN a correctly instantiated trainer
    GIVEN some training has been done
        THEN there should not be any logs inside the default log folder
        THEN there exists a logdir folder for each metric
        THEN there inside in each folder there's the JSON file w/ the metric logs
        THEN in the file there are the correct keys

    """
    training_completed, trainer = training_loop(
        logdir=tmpdir, metrics=metrics, **loop_args
    )

    assert training_completed
    assert not pathlib.Path(DEFAULT_LOGDIR).exists()  # Assert absence of side effects
    # Assert there exists folder for each metric
    for metric in trainer._metrics:
        metric_dir = pathlib.Path(tmpdir).joinpath(
            "best", metric.name.replace("/", "_")
        )
        assert metric_dir.exists()
        json_path = metric_dir.joinpath(f"{metric.name.replace('/', '_')}.json")
        assert json_path.exists()
        with open(json_path, "r") as fp:
            metric_data = json.load(fp)

            # Assert the metric data contains the expected keys
            assert metric.name.replace("/", "_") in metric_data
            assert "step" in metric_data


# -------------------------------------------------------------------------------------


TEST_MATRIX_MODEL_SELECTION = {
    # TODO: Add test for a metric with operator.gt
    "classifier_trainer_lt": [
        fake_classifier_taining_loop,
        {},
        [ClassifierLoss(model_selection_operator=operator.lt)],
        operator.lt,
    ],
}
TEST_PARAMS_MODEL_SELECTION = [
    TEST_MATRIX_MODEL_SELECTION[k] for k in TEST_MATRIX_MODEL_SELECTION
]
TEST_IDS_MODEL_SELECTION = [k for k in TEST_MATRIX_MODEL_SELECTION]


@pytest.mark.parametrize(
    ["training_loop", "loop_args", "metrics", "operator_check"],
    TEST_PARAMS_MODEL_SELECTION,
    ids=TEST_IDS_MODEL_SELECTION,
)
def test_model_selection(
    training_loop, loop_args, metrics, operator_check: List[Metric]
):
    """
    Test the correct model selection behaviour of metrics.

    Model selection is handled by the Metric when triggered by a trainer.

    GIVEN a correctly instantiated trainer
    GIVEN some training has been done
        THEN there should be a metric log file containing two values
        GIVEN all metrics log get initialized at -inf at step 0
        GIVEN a new data point is added to the log
        WHEN performing model solection this should be the value used

    """
    number_of_metrics = len(metrics)
    for metric in metrics:
        assert metric._model_selection_operator == operator_check

    training_completed, trainer = training_loop(
        logdir="testlog", metrics=metrics, **loop_args
    )

    # Maybe make it explicit when a trainer popultate metrics autonomously
    assert training_completed
    # Manually have to check this in the log. Find a better way.
    # loss: validation value: inf → 0.0


# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("training_loop", [fake_classifier_taining_loop])
def test_metrics_names_collision(training_loop, tmpdir):
    """
    Test that an exception is correctly raised when two metrics have the same name.

    WHEN two or more metrics passed to a trainer have the same name
        THEN raise a ValueError
    """
    metrics = [ClassifierLoss(name="test_loss"), ClassifierLoss(name="test_loss")]
    with pytest.raises(ValueError):
        training_completed, trainer = training_loop(metrics=metrics, logdir=tmpdir)
