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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import numpy as np
import tensorflow as tf  # pylint: disable=import-error

if TYPE_CHECKING:
    from ashpy.contexts import Context

__ALL__ = ["Metric"]


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
        logdir: Union[Path, str] = Path.cwd() / "log",
    ) -> None:
        """
        Initialize the Metric object.

        Args:
            name (str): Name of the metric.
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
        self._logdir = Path(logdir) if not isinstance(logdir, Path) else logdir

    def model_selection(
        self, checkpoint: tf.train.Checkpoint, global_step: tf.Variable
    ) -> Optional[Path]:
        """
        Perform model selection.

        Args:
            checkpoint (:py:class:`tf.train.Checkpoint`): Checkpoint object that contains
                the model status.
            global_step (:py:class:`tf.Variable`): current training step

        """
        current_value = self.result()
        previous_value = float(
            self.json_read(self.best_model_sel_file)[self.sanitized_name]
        )
        # Model selection is done ONLY if an operator was passed at __init__
        if self._model_selection_operator and self._model_selection_operator(
            current_value, previous_value
        ):
            tf.print(
                f"{self.sanitized_name}: validation value: {previous_value} → {current_value}"
            )
            self.json_write(
                self.best_model_sel_file,
                {
                    self.sanitized_name: str(current_value),
                    "step": int(global_step.numpy()),
                },
            )
            manager = tf.train.CheckpointManager(
                checkpoint, self.best_folder / "ckpts", max_to_keep=1
            )
            return Path(manager.save())
        return None

    def _update_logdir(self):
        if not self._model_selection_operator:
            pass
        # write the initial value of the best metric
        if not self.best_model_sel_file.exists():
            self.best_model_sel_file.parent.mkdir(parents=True)
        initial_value = (
            np.inf if self._model_selection_operator is operator.lt else -np.inf
        )
        self.json_write(
            self.best_model_sel_file,
            {self.sanitized_name: str(initial_value), "step": 0},
        )

    @property
    def name(self) -> str:
        """Retrieve the metric name."""
        return self._name

    @property
    def sanitized_name(self) -> str:
        """
        Retrieve the sanitized name: all / are _.

        This is done since adding a prefix to a metric name with a / allows for TensorBoard
        automatic grouping. When we are not working with TB we want to replace all / with _.
        """
        return self._name.replace("/", "_")

    @property
    def metric(self) -> tf.keras.metrics.Metric:
        """Retrieve the :py:class:`tf.keras.metrics.Metric` object."""
        return self._metric

    @property
    def model_selection_operator(self) -> Optional[Callable]:
        """Retrieve the operator used for model selection."""
        return self._model_selection_operator

    @property
    def logdir(self) -> Path:
        """Retrieve the log directory."""
        return self._logdir

    @logdir.setter
    def logdir(self, logdir) -> None:
        """Set the logdir changing also other properties."""
        self._logdir = logdir
        self._update_logdir()

    @property
    def best_folder(self) -> Path:
        """Retrieve the folder used to save the best model when doing model selection."""
        return self.logdir / "best" / self.sanitized_name

    @property
    def best_model_sel_file(self) -> Path:
        """Retrieve the path to JSON file containing the measured performance of the best model."""
        return self.best_folder / (self.sanitized_name + ".json")

    @staticmethod
    def json_read(filename: Path) -> Dict[str, Any]:
        """
        Read a JSON file.

        Args:
            filename (str): The path to the JSON file to read.

        Returns:
            :py:obj:`typing.Dict`: Dictionary containing the content of the JSON file.

        """
        if not filename.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        data: Dict[str, Union[str, int, float]] = {}
        with open(filename, "r") as fp:
            data = json.load(fp)

        return data

    @staticmethod
    def json_write(filename: Path, what_to_write: Dict) -> None:
        """
        Write inside the specified JSON file the mean and stddev.

        Args:
            filename (str): Path to the JSON file to write.
            what_to_write (dict): A dictionary containing what to write.

        """
        if filename.exists():
            data = Metric.json_read(filename)
            for key in what_to_write:
                data[key] = str(what_to_write[key])
        else:
            data = what_to_write
            if not filename.parent.exists():
                filename.parent.mkdir()

        with open(filename, "w+") as fp:
            json.dump(data, fp, indent=4)

    @abstractmethod
    def update_state(self, context: Context) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.Context`): An AshPy Context holding
                all the information the Metric needs.

        """

    def result(self):
        """
        Get the result of the metric.

        Returns:
            :py:class:`numpy.ndarray`: The current value of the metric.

        """
        return self._metric.result().numpy()

    def log(self, step: int) -> None:
        """
        Log the metric.

        Args:
            step: global step of training

        """
        tf.summary.scalar(self.name, self.result(), step=step)

    def reset_states(self) -> None:
        """Reset the state of the metric."""
        self._metric.reset_states()
