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

from typing import List, Optional

import tensorflow as tf
from ashpy.metrics import Metric
from ashpy.modes import LogEvalMode


class Context:
    """:py:class:`ashpy.contexts.Context` provide an interface for all contexts."""

    def __init__(
        self,
        metrics: List[Metric] = None,
        dataset: tf.data.Dataset = None,
        log_eval_mode: LogEvalMode = LogEvalMode.TEST,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        checkpoint: tf.train.Checkpoint = None,
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
            checkpoint (:py:class:`tf.train.Checkpoint`): Checkpoint to use to keep track of
                models status.

        """
        self._distribute_strategy = tf.distribute.get_strategy()

        # TODO: are metrics really needed right now?
        self._metrics = metrics if metrics else []
        self._dataset = dataset
        self._log_eval_mode = log_eval_mode
        self._global_step = global_step
        self._checkpoint = checkpoint
        self._exception: Optional[Exception] = None
        self._current_batch: Optional[tf.Tensor] = None

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
            :py:class:`tf.data.Dataset` the current dataset

        """
        return self._dataset

    @dataset.setter
    def dataset(self, _dataset: tf.data.Dataset):
        """
        Set the dataset.

        Args:
            _dataset (:py:class:`tf.data.Dataset`): dataset to set

        """
        self._dataset = _dataset

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

    @property
    def exception(self) -> Optional[Exception]:
        """Return the exception."""
        return self._exception

    @exception.setter
    def exception(self, exception: Optional[Exception]) -> None:
        """Set the exception."""
        self._exception = exception

    @property
    def current_batch(self) -> Optional[tf.Tensor]:
        """Return the current batch."""
        return self._current_batch

    @current_batch.setter
    def current_batch(self, _current_batch: Optional[tf.Tensor]) -> None:
        """Set the current_batch."""
        self._current_batch = _current_batch
