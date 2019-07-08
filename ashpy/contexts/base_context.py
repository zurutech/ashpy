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

r"""
Primitive Context Interface.

``Contexts`` are checkpointable (subclassed from :py:class:`tf.train.Checkpoint`)
collections of variable encapsulated in a Python Class as a way to seamlessly
handle information transfer.
"""

import tensorflow as tf
from ashpy.metrics import Metric
from ashpy.modes import LogEvalMode


class BaseContext:
    r"""
    :py:class:`ashpy.contexts.base_context.BaseContext` provide an interface for all contexts to inherit from.
    """

    def __init__(
        self,
        metrics=None,
        dataset=None,
        log_eval_mode=LogEvalMode.TEST,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        ckpt=None,
    ):
        r"""
        :py:class:`ashpy.contexts.base_context.BaseContext`

        Args:
            metrics ([:py:class:`ashpy.metrics.metric.Metric`]): array of :py:class:`ashpy.metrics.metric.Metric` objects.
            dataset (:py:class:`tf.data.Dataset`): The dataset to use, that
                contains everything needed to use the model in this context.
            log_eval_mode: models' mode to use when evaluating and logging.
            global_step: tf.Variable that keeps track of the training steps.
            ckpt (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of models status.
        """
        self._distribute_strategy = tf.distribute.get_strategy()
        self._metrics = metrics if metrics else []
        self._validate_metrics()
        self._dataset = dataset
        self._log_eval_mode = log_eval_mode
        self._global_step = global_step
        self._ckpt = ckpt

    def _validate_metrics(self):
        """Check if every metric is an :py:class:`ashpy.metrics.Metric`."""
        for metric in self._metrics:
            if not isinstance(metric, Metric):
                raise ValueError(
                    "Metric " + str(metric) + " is not a ashpy.metrics.Metric"
                )

    def measure_metrics(self):
        """Measure the metrics."""
        for metric in self._metrics:
            metric.update_state(self)

    def model_selection(self):
        """Use the metrics to perform model selection."""
        for metric in self._metrics:
            metric.model_selection(self._ckpt)

    @property
    def log_eval_mode(self):
        """Model(s) mode."""
        return self._log_eval_mode

    @property
    def dataset(self):
        """Return dataset."""
        return self._dataset

    @property
    def metrics(self):
        """Return the metrics."""
        return self._metrics

    @property
    def global_step(self):
        """Return the global_step."""
        return self._global_step
