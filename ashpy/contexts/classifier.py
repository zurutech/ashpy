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

"""Classifier Context."""
import tensorflow as tf

from ashpy.contexts.base_context import BaseContext
from ashpy.modes import LogEvalMode


class ClassifierContext(BaseContext):
    r""":py:class:`ashpy.contexts.classifier.ClassifierContext` provide the standard functions to test a classifier."""

    def __init__(
        self,
        classifier_model=None,
        loss=None,
        dataset=None,
        metrics=None,
        log_eval_mode=LogEvalMode.TEST,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        ckpt=None,
    ):
        r"""
        Instantiate the :py:class:`ashpy.contexts.classifier.ClassifierContext` context.

        Args:
            classifier_model (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model`
                model.
            loss (callable): loss function, format f(y_true, y_pred)
            dataset (:py:class:`tf.data.Dataset`): The test dataset.
            metrics: List of python objects (of Metric class) with which to measure
                training and validation data performances.
            log_eval_mode: models' mode to  use when evaluating and logging.
            global_step: tf.Variable that keeps track of the training steps.
            ckpt (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of models status.
        """
        super().__init__(metrics, dataset, log_eval_mode, global_step, ckpt)
        self._classifier_model = classifier_model
        self._loss = loss

    @property
    def loss(self):
        """Return the loss value."""
        return self._loss

    @property
    def classifier_model(self):
        r"""
        Return the Model Object.

        Returns:
            :py:class:`tf.keras.Model`

        """
        return self._classifier_model
