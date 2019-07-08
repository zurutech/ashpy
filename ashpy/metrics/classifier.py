#Copyright 2019 Zuru Tech HK Limited. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""The classification metrics."""

import os

import tensorflow as tf

from ashpy.metrics.metric import Metric
from ashpy.modes import LogEvalMode


class ClassifierLoss(Metric):
    """A handy way to measure the classification loss."""

    def __init__(
        self, model_selection_operator=None, logdir=os.path.join(os.getcwd(), "log")
    ):
        """
        Args:
            model_selection_operator: The operation to be used when model_selection
                is on to compare the metrics. E.g.: operator.gt.
            logdir: The log dir.

        Returns:
            :obj:`None`.
        """

        super().__init__(
            name="loss",
            metric=tf.metrics.Mean(name="loss", dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        return self._metric.reset_states()

    def update_state(self, context):
        for features, labels in context.dataset:
            loss = context.loss(
                context,
                features=features,
                labels=labels,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )
            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(loss)
            )


class ClassifierMetric(Metric):
    """A wrapper for the classification metric that requires to apply argmax
    to the classifier output to extract the predictions."""

    def __init__(
        self,
        metric,
        model_selection_operator=None,
        logdir=os.path.join(os.getcwd(), "log"),
        processing_predictions={"fn": tf.argmax, "kwargs": {"axis": -1}},
    ):
        """
        Args:
            metric: The metric for the classifier (e.g.: Accuracy()).
            model_selection_operator: The operation, if needed, to be used
                to confront the metric value (e.g.: operator.gt).
            logdir: The logdir.
            processing_predictions: A dict in the form of
                {"fn": tf.argmax, "kwargs": {"axis": -1}} with a function "fn"
                to be used for predictions processing purposes and its "kwargs" as an inner dict.

        Returns:
            :obj:`None`.

        """

        super().__init__(
            name=metric.name,
            metric=metric,
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )
        self._processing_predictions = processing_predictions

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        self._metric.reset_states()

    def update_state(self, context):
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
