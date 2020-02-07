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
import shutil
from pathlib import Path

import pytest
from ashpy.metrics import ClassifierLoss

from tests.utils.fake_training_loop import FakeAdversarialTraining, FakeTraining

DEFAULT_LOGDIR = "log"
OPERATOR_INITIAL_VALUE_MAP = {operator.gt: "-inf", operator.lt: "inf"}


@pytest.fixture(scope="module")
def cleanup():
    """Remove the default logdir before and after testing."""
    default_log_dir = Path(DEFAULT_LOGDIR)
    if default_log_dir.exists():
        shutil.rmtree(default_log_dir)
    yield "Cleanup"
    if default_log_dir.exists():
        shutil.rmtree(default_log_dir)


def get_metric_data(metric, tmpdir):
    """Make sure that the metric exists and is readable. Return its value."""
    metric_dir = Path(tmpdir) / "best" / metric.sanitized_name
    assert metric_dir.exists()
    json_path = metric_dir / f"{metric.sanitized_name}.json"
    assert json_path.exists()
    with open(json_path, "r") as fp:
        metric_data = json.load(fp)

        # Assert the metric data contains the expected keys
        assert metric.sanitized_name in metric_data
        assert "step" in metric_data
        return metric_data[metric.sanitized_name]


def test_metrics_log(fake_training_fn, tmpdir, cleanup):
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
    logdir = Path(tmpdir)

    fake_training: FakeTraining = fake_training_fn(logdir=logdir)
    assert fake_training()
    trainer = fake_training.trainer

    assert not Path(DEFAULT_LOGDIR).exists()  # Assert absence of side effects
    # Assert there exists folder for each metric
    for metric in trainer._metrics:

        metric_value = get_metric_data(metric, tmpdir)
        # Assert that the correct model selection has been performed
        # Check it by seeing if the values in the json has been updated
        if metric.model_selection_operator:
            try:
                initial_value = OPERATOR_INITIAL_VALUE_MAP[
                    metric.model_selection_operator
                ]
            except KeyError:
                raise ValueError(
                    "Please add the initial value for this "
                    "operator to OPERATOR_INITIAL_VALUE_MAP"
                )

            assert metric_value != initial_value


# -------------------------------------------------------------------------------------


def test_metrics_names_collision(tmpdir):
    """
    Test that an exception is correctly raised when two metrics have the same name.

    WHEN two or more metrics passed to a trainer have the same name
        THEN raise a ValueError
    """
    metrics = [
        ClassifierLoss(name="test_loss"),
        ClassifierLoss(name="test_loss"),
    ]

    with pytest.raises(ValueError):
        # Since names collision is checked by the primitive Trainer we can test it easily
        # with just one FakeTraining and be assured that it works.
        FakeAdversarialTraining(tmpdir, metrics=metrics)


# -------------------------------------------------------------------------------------


# def test_metrics_on_restart(fake_training, tmpdir):
#     """Test that metrics are correctly read on train restart."""

#     def _train(fake_training, tmpdir):
#         training_loop, loop_args, metrics = fake_training
#         training_completed, trainer = training_loop(
#             logdir=tmpdir, metrics=metrics, **loop_args
#         )
#         assert training_completed
#         # Read and store the values of the metrics
#         metrics_values = {
#             metric.name: get_metric_data(metric, tmpdir) for metric in trainer._metrics
#         }
#         return metrics_values

#     t1_values = _train(fake_training, tmpdir)
#     t2_values = _train(fake_training, tmpdir)
#     assert t1_values == t2_values
