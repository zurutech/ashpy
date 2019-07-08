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

"""The classification losses."""

import tensorflow as tf
from ashpy.losses.executor import Executor


class ClassifierLoss(Executor):
    r"""
    Classifier Loss Executor using the classifier model, instantiated with a fn
    """

    def __init__(self, fn):
        r"""
        Constructor of :py:class:`ClassifierLoss`
        Args:
            fn: Classification Loss function, should take as input labels and prediction
        """
        super().__init__(fn)

    @Executor.reduce_loss
    def call(self, context, *, features, labels, training, **kwargs):
        r"""
        Compute the classifier loss
        Args:
            context (:py:class:`ashpy.contexts.classifier.ClassifierContext`): Context for classification
            features: inputs for the classifier model
            labels: target for the classifier model
            training: whether is training or not
            **kwargs:

        Returns:
            :py:class:`tf.Tensor` loss value

        """
        predictions = context.classifier_model(features, training=training)
        loss = self._fn(labels, predictions)
        loss = tf.cond(
            tf.equal(tf.rank(loss), tf.constant(4)),
            lambda: loss,
            lambda: tf.expand_dims(tf.expand_dims(loss, axis=-1), axis=-1),
        )
        return tf.reduce_mean(loss, axis=[1, 2])
