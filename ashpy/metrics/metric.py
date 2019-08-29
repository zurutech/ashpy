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

"""Metric is the abstract class that every ash metric must implement."""

from __future__ import annotations

import errno
import json
import operator
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import numpy as np
import tensorflow as tf  # pylint: disable=import-error

if TYPE_CHECKING:
    from ashpy.contexts import BaseContext


class Metric(ABC):
    """
    Metric is the abstract class that every ash Metric must implement.

    AshPy Metrics wrap and extend Keras Metrics.
    """

    def __init__(
        self,
        name: str,
        metric: tf.keras.metrics.Metric,
        model_selection_operator: Callable = None,
        logdir: str = os.path.join(os.getcwd(), "log"),
    ) -> None:
        """
        Initialize the Metric object.

        Args:
            name (str): The name of the metric.
            metric (:py:class:`tf.keras.metrics.Metric`): The Keras metric to use.
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an `model_selection_operator` is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.

        """
        self._distribute_strategy = tf.distribute.get_strategy()
        self._name = name
        self._metric = metric
        self._model_selection_operator = model_selection_operator
        self.logdir = logdir

    def model_selection(
        self, checkpoint: tf.train.Checkpoint, global_step: tf.Variable
    ) -> None:
        """
        Perform model selection.

        Args:
            checkpoint (:py:class:`tf.train.Checkpoint`): Checkpoint object that contains
                the model status.
            global_step (:py:class:`tf.Variable`): current training step

        """
        current_value = self.result()
        previous_value = float(self.json_read(self.best_model_sel_file)[self._name])
        # Model selection is done ONLY if an operator was passed at __init__
        if self._model_selection_operator and self._model_selection_operator(
            current_value, previous_value
        ):
            tf.print(
                f"{self.name}: validation value: {previous_value} → {current_value}"
            )
            Metric.json_write(
                self.best_model_sel_file,
                {self._name: str(current_value), "step": int(global_step.numpy())},
            )
            manager = tf.train.CheckpointManager(
                checkpoint, os.path.join(self.best_folder, "ckpts"), max_to_keep=1
            )
            manager.save()

    @property
    def name(self) -> str:
        """Retrieve the metric name."""
        return self._name

    @property
    def metric(self) -> tf.keras.metrics.Metric:
        """Retrieve the :py:class:`tf.keras.metrics.Metric` object."""
        return self._metric

    @property
    def model_selection_operator(self) -> Optional[Callable]:
        """Retrieve the operator used for model selection."""
        return self._model_selection_operator

    @property
    def logdir(self) -> str:
        """Retrieve the log directory."""
        return self._logdir

    @logdir.setter
    def logdir(self, logdir) -> None:
        """Set the logdir changing also other properties."""
        self._logdir = logdir

        # write the initial value of the best metric
        if not os.path.exists(self.best_model_sel_file):
            os.makedirs(os.path.dirname(self.best_model_sel_file))
        initial_value = (
            np.inf if self._model_selection_operator is operator.lt else -np.inf
        )
        self.json_write(
            self.best_model_sel_file, {self._name: str(initial_value), "step": 0}
        )

    @property
    def best_folder(self) -> str:
        """Retrieve the folder used to save the best model when doing model selection."""
        return os.path.join(self.logdir, "best", self._name)

    @property
    def best_model_sel_file(self) -> str:
        """Retrieve the path to JSON file containing the measured performance of the best model."""
        return os.path.join(self.best_folder, self._name + ".json")

    @staticmethod
    def json_read(filename: str) -> Dict[str, Any]:
        """
        Read a JSON file.

        Args:
            filename (str): The path to the JSON file to read.

        Returns:
            :py:obj:`typing.Dict`: Dictionary containing the content of the JSON file.

        """
        if not os.path.exists(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        data: Dict[str, Union[str, int, float]] = {}
        with open(filename, "r") as fp:
            data = json.load(fp)

        return data

    @staticmethod
    def json_write(filename: str, what_to_write: Dict) -> None:
        """
        Write inside the specified JSON file the mean and stddev.

        Args:
            filename (str): Path to the JSON file to write.
            what_to_write (dict): A dictionary containing what to write.

        """
        if os.path.exists(filename):
            data = Metric.json_read(filename)
            for key in what_to_write:
                data[key] = str(what_to_write[key])
        else:
            data = what_to_write
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

        with open(filename, "w+") as fp:
            json.dump(data, fp, indent=4)

    @abstractmethod
    def update_state(self, context: BaseContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.BaseContext`): An AshPy Context holding
                all the information the Metric needs.

        """

    def result(self):
        """
        Get the result of the metric.

        Returns:
            :py:class:`numpy.ndarray`: The current value of the metric.

        """
        return self._metric.result().numpy()

    def log(self, step) -> None:
        """
        Log the metric
        Args:
            step: global step of training
        """
        tf.summary.scalar(self.name, self.result(), step=step)

    def reset_states(self) -> None:
        """Reset the state of the metric."""
        self._metric.reset_states()
