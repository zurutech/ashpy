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

TODO: Adversarial Encoder Traner
"""
import json
import operator
import pathlib
import shutil

import pytest
from ashpy.metrics import ClassifierLoss

from tests.utils.fake_training_loop import fake_classifier_training_loop

DEFAULT_LOGDIR = "log"
OPERATOR_INITIAL_VALUE_MAP = {operator.gt: "-inf", operator.lt: "inf"}


@pytest.fixture(scope="module")
def cleanup():
    """Remove the default logdir before and after testing."""
    default_log_dir = pathlib.Path(DEFAULT_LOGDIR)
    if default_log_dir.exists():
        shutil.rmtree(default_log_dir)
    yield "Cleanup"
    if default_log_dir.exists():
        shutil.rmtree(default_log_dir)


def test_metrics_log(fake_training, tmpdir, cleanup):
    """
    Test that trainers correctly create metrics log files.

    Also test that model selection has been correctly performed.

    GIVEN a correctly instantiated trainer
    GIVEN some training has been done
        THEN there should not be any logs inside the default log folder
        THEN there exists a logdir folder for each metric
        THEN there inside in each folder there's the JSON file w/ the metric logs
        THEN in the file there are the correct keys'
        THEN the values of the keys should not be the operator initial value

    """
    training_loop, loop_args, metrics = fake_training

    training_completed, trainer = training_loop(
        logdir=tmpdir, metrics=metrics, **loop_args
    )

    assert training_completed
    assert not pathlib.Path(DEFAULT_LOGDIR).exists()  # Assert absence of side effects
    # Assert there exists folder for each metric
    for metric in trainer._metrics:
        metric_dir = pathlib.Path(tmpdir) / "best" / metric.sanitized_name
        assert metric_dir.exists()
        json_path = metric_dir / f"{metric.sanitized_name}.json"
        assert json_path.exists()
        with open(json_path, "r") as fp:
            metric_data = json.load(fp)

            # Assert the metric data contains the expected keys
            assert metric.sanitized_name in metric_data
            assert "step" in metric_data

            # Assert that the correct model selection has been performed
            # Check it by seeing if the values in the json has been updated
            if metric.model_selection_operator:
                try:
                    initial_value = OPERATOR_INITIAL_VALUE_MAP[
                        metric.model_selection_operator
                    ]
                except KeyError:
                    raise ValueError(
                        "Please add the initial value for this operator to OPERATOR_INITIAL_VALUE_MAP"
                    )
                assert metric_data[metric.sanitized_name] != initial_value


# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("training_loop", [fake_classifier_training_loop])
def test_metrics_names_collision(training_loop, tmpdir):
    """
    Test that an exception is correctly raised when two metrics have the same name.

    WHEN two or more metrics passed to a trainer have the same name
        THEN raise a ValueError
    """
    metrics = [ClassifierLoss(name="test_loss"), ClassifierLoss(name="test_loss")]
    with pytest.raises(ValueError):
        training_loop(metrics=metrics, logdir=tmpdir)
