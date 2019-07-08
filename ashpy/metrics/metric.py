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

import errno
import json
import operator
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Metric(ABC):
    """Metric is the abstract class that every ash metric must implement."""

    def __init__(
        self,
        name,
        metric,
        model_selection_operator=None,
        logdir=os.path.join(os.getcwd(), "log"),
    ):
        """
        Construct the Metric object.

        Args:
            name: the name of the metric.
            metric: The metric. This must be a keras metric.
            model_selection_operator: The operation to be used when model_selection
                is on to compare the metrics. E.g.: operator.gt.
            feeding it to the metric update_state function.

        Returns:
            :obj:`None`.
        """

        self._distribute_strategy = tf.distribute.get_strategy()
        self._name = name
        self._metric = metric
        self._model_selection_operator = model_selection_operator
        self.logdir = logdir

    def model_selection(self, checkpoint):
        """Perform model selection.
        Args:
            context: the checkpoint object that contains the model status.
        """
        current_value = self.result()
        previous_value = float(self.json_read(self.best_model_sel_file)[self._name])
        if self._model_selection_operator and self._model_selection_operator(
            current_value, previous_value
        ):
            tf.print(
                f"{self.name}: validation value: {previous_value} → {current_value}"
            )
            Metric.json_write(
                self.best_model_sel_file, {self._name: str(current_value)}
            )
            manager = tf.train.CheckpointManager(
                checkpoint, os.path.join(self.best_folder, "ckpts"), max_to_keep=1
            )
            manager.save()

    @property
    def name(self):
        """The metric name."""
        return self._name

    @property
    def metric(self):
        """The keras metric object."""
        return self._metric

    @property
    def model_selection_operator(self):
        """The model selection operator."""
        return self._model_selection_operator

    @property
    def logdir(self):
        """The log directory."""
        return self._logdir

    @logdir.setter
    def logdir(self, logdir):
        """Setting the logdir changes also other properties."""
        self._logdir = logdir

        # write the initial value of the best metric
        if not os.path.exists(self.best_model_sel_file):
            os.makedirs(os.path.dirname(self.best_model_sel_file))
        initial_value = (
            np.inf if self._model_selection_operator is operator.lt else -np.inf
        )
        self.json_write(self.best_model_sel_file, {self._name: str(initial_value)})

    @property
    def best_folder(self):
        """The folder to use to save the best model when doing model selection."""
        return os.path.join(self.logdir, "best", self._name)

    @property
    def best_model_sel_file(self):
        """The JSON file that contains the measured performance of the best model."""
        return os.path.join(self.best_folder, self._name + ".json")

    @staticmethod
    def json_read(filename):
        """
        Read a Json file.
        Args:
            filename: The JSON file to read.

        Returns:
            data: The data read.
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        data = {}
        with open(filename, "r") as fp:
            data = json.load(fp)

        return data

    @staticmethod
    def json_write(filename, what_to_write):
        """
        Write inside the JSON file specified the mean and stddev.

        Args:
            filename: The JSON file to write.
            what_to_write: A dict containing what to write

        Returns:
             :obj:`None`.
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
    def update_state(self, context):
        """
        Update the internal state of the metric, using
        the information from the context object.
        Args:
            context: a Context Object that carries all the metric needs.
        """

    @abstractmethod
    def result(self):
        """
        Get the result of the metric.

        Returns:
            The current value of the metric.
        """

    @abstractmethod
    def reset_states(self):
        """
        Reset the state of the metric.

        Returns:
             :obj:`None`.
        """
