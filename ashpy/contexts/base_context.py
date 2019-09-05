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

from typing import List

import tensorflow as tf

from ashpy.metrics import Metric
from ashpy.modes import LogEvalMode


class BaseContext:
    """:py:class:`ashpy.contexts.BaseContext` provide an interface for all contexts."""

    def __init__(
        self,
        metrics: List[Metric] = None,
        dataset: tf.data.Dataset = None,
        log_eval_mode: LogEvalMode = LogEvalMode.TEST,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        ckpt: tf.train.Checkpoint = None,
    ) -> None:
        """
        Initialize the Context.

        Args:
            metrics (:obj:`list` of [:py:class:`ashpy.metrics.metric.Metric`]): List of
                :py:class:`ashpy.metrics.metric.Metric` objects.
            dataset (:py:class:`tf.data.Dataset`): The dataset to use, that
                contains everything needed to use the model in this context.
            log_eval_mode (:py:class:`ashpy.modes.LogEvalMode`): Models' mode to use when
                evaluating and logging.
            global_step (:py:class:`tf.Variable`): Keeps track of the training steps.
            ckpt (:py:class:`tf.train.Checkpoint`): Checkpoint to use to keep track of
                models status.

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

    def measure_metrics(self) -> None:
        """Measure the metrics."""
        for metric in self._metrics:
            metric.update_state(self)

    def model_selection(self) -> None:
        """Use the metrics to perform model selection."""
        for metric in self._metrics:
            metric.model_selection(self._ckpt, self._global_step)

    @property
    def log_eval_mode(self) -> LogEvalMode:
        """
        Retrieve model(s) mode.

        Returns:
            :py:class:`ashpy.modes.LogEvalMode`.

        """
        return self._log_eval_mode

    @property
    def dataset(self) -> tf.data.Dataset:
        """
        Retrieve the dataset.

        Returns:
            :py:class:`tf.data.Dataset`.

        """
        return self._dataset

    @property
    def metrics(self) -> List[Metric]:
        """
        Retrieve the metrics.

        Returns:
            :obj:`list` of [:py:class:`ashpy.metrics.metric.Metric`].

        """
        return self._metrics

    @property
    def global_step(self) -> tf.Variable:
        """
        Retrieve the global_step.

        Returns:
            :py:class:`tf.Variable`.

        """
        return self._global_step
