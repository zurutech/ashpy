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

"""The classification metrics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

import tensorflow as tf  # pylint: disable=import-error
from ashpy.metrics.metric import Metric
from ashpy.modes import LogEvalMode

if TYPE_CHECKING:
    from ashpy.contexts import ClassifierContext  # pylint: disable=ungrouped-imports

    TPRocessingPredictions = Dict[str, Union[Callable, Dict[str, Any]]]

__ALL__ = ["ClassifierLoss", "ClassifierMetric"]


class ClassifierLoss(Metric):
    """A handy way to measure the classification loss."""

    def __init__(
        self,
        name: str = "loss",
        model_selection_operator: Callable = None,
        logdir: Union[Path, str] = Path().cwd() / "log",
    ) -> None:
        """
        Initialize the Metric.

        Args:
            name (str): Name of the metric.
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an operator is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.

        """
        super().__init__(
            name=name,
            metric=tf.metrics.Mean(name=name, dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

    def update_state(self, context: ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context
                holding all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        for features, labels in context.dataset:
            loss = context.loss(
                context,
                features=features,
                labels=labels,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )
            self._distribute_strategy.experimental_run_v2(updater(loss))


class ClassifierMetric(Metric):
    """Wrap a metric using `argmax` to extract predictions out of a classifier's output."""

    def __init__(
        self,
        metric: tf.keras.metrics.Metric,
        model_selection_operator: Callable = None,
        logdir: Union[Path, str] = Path().cwd() / "log",
        processing_predictions=None,
    ) -> None:
        """
        Initialize the Metric.

        Args:
            metric (:py:class:`tf.keras.metrics.Metric`): The Keras Metric to use with
                the classifier (e.g.: Accuracy()).
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an `model_selection_operator` is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.
            processing_predictions (:py:obj:`typing.Dict`): A `dict` in the form of
                `{"fn": tf.argmax, "kwargs": {"axis": -1}}` with a function `"fn"`
                to be used for predictions processing purposes and its `"kwargs"` as its
                keyword-arguments. Defaults to {"fn": tf.argmax, "kwargs": {"axis": -1}}.

        """
        super().__init__(
            name=metric.name,
            metric=metric,
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )
        if processing_predictions is None:
            processing_predictions = {"fn": tf.argmax, "kwargs": {"axis": -1}}
        self._processing_predictions = processing_predictions

    def update_state(self, context: ClassifierContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context holding
                all the information the Metric needs.

        """
        for features, labels in context.dataset:
            predictions = context.classifier_model(
                features, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(
                    labels,
                    self._processing_predictions["fn"](
                        predictions, **self._processing_predictions["kwargs"]
                    ),
                )
            )
