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

"""GAN metrics."""
from __future__ import annotations

import operator
import os
import types
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import tensorflow as tf
import tensorflow_hub as hub
from ashpy.metrics import ClassifierMetric, Metric
from ashpy.modes import LogEvalMode

if TYPE_CHECKING:
    from ashpy.contexts import (  # pylint: disable=ungrouped-imports
        GANContext,
        GANEncoderContext,
    )

__ALL__ = [
    "DiscriminatorLoss",
    "EncoderLoss",
    "EncodingAccuracy",
    "GeneratorLoss",
    "InceptionScore",
]


class DiscriminatorLoss(Metric):
    """The Discriminator loss value."""

    def __init__(
        self,
        name: str = "d_loss",
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

    def update_state(self, context: GANContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANContext`): An AshPy Context Object
                that carries all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)

        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy

            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )
            loss = context.discriminator_loss(
                context,
                fake=fake,
                real=real_x,
                condition=real_y,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )

            self._distribute_strategy.experimental_run_v2(updater(loss))


class GeneratorLoss(Metric):
    """Generator loss value."""

    def __init__(
        self,
        name: str = "g_loss",
        model_selection_operator: Callable = None,
        logdir: Union[Path, str] = Path().cwd() / "log",
    ):
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

    def update_state(self, context: GANContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.GANContext`): An AshPy Context Object that carries
                all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy
            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            loss = context.generator_loss(
                context,
                fake=fake,
                real=real_x,
                condition=real_y,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )

            self._distribute_strategy.experimental_run_v2(updater(loss))


class EncoderLoss(Metric):
    """Encoder Loss value."""

    def __init__(
        self,
        name: str = "e_loss",
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

    def update_state(self, context: GANEncoderContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANEncoderContext`): An AshPy Context Object
                that carries all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy
            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]
            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            loss = context.encoder_loss(
                context,
                fake=fake,
                real=real_x,
                condition=real_y,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )

            self._distribute_strategy.experimental_run_v2(updater(loss))


class InceptionScore(Metric):
    r"""
    Inception Score Metric.

    This class is an implementation of the Inception Score technique for evaluating a GAN.

    See Improved Techniques for Training GANs [1]_.

    .. [1] Improved Techniques for Training GANs https://arxiv.org/abs/1606.03498

    """

    def __init__(
        self,
        inception: tf.keras.Model,
        name: str = "inception_score",
        model_selection_operator=operator.gt,
        logdir=Path().cwd() / "log",
    ):
        """
        Initialize the Metric.

        Args:
            inception (:py:class:`tf.keras.Model`): Keras Inception model.
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
            metric=tf.metrics.Mean(name),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._inception_model = inception

        # add softmax layer if not present
        if "softmax" not in self._inception_model.layers[-1].name.lower():
            self._inception_model = tf.keras.Sequential(
                [self._inception_model, tf.keras.layers.Softmax()]
            )

    def update_state(self, context: GANContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context
                holding all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)

        # Generate the images created with the AshPy Context's generator
        for real_xy, noise in context.dataset:
            _, real_y = real_xy

            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            # rescale images between 0 and 1
            fake = (fake + 1.0) / 2.0

            # Resize images to 299x299
            fake = tf.image.resize(fake, (299, 299))

            try:
                fake = tf.image.grayscale_to_rgb(fake)
            except ValueError:
                # Images are already RGB
                pass

            # Calculate the inception score
            inception_score_per_batch = self.inception_score(fake)

            # Update the Mean metric created for this context
            # self._metric.update_state(mean)
            self._distribute_strategy.experimental_run_v2(
                updater(inception_score_per_batch)
            )

    def inception_score(self, images: tf.Tensor) -> tf.Tensor:
        """
        Compute the Inception Score.

        Args:
            images (:py:obj:`list` of [:py:class:`numpy.ndarray`]): A list of ndarray of
                generated images of 299x299 of size.

        Returns:
            :obj:`tuple` of (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`): Mean and STD.

        """
        tf.print("Computing inception score...")

        predictions: tf.Tensor = self._inception_model(images)

        kl_divergence = predictions * (
            tf.math.log(predictions)
            - tf.math.log(tf.math.reduce_mean(predictions, axis=0, keepdims=True))
        )
        kl_divergence = tf.math.reduce_mean(tf.math.reduce_sum(kl_divergence, axis=1))
        inception_score_per_batch = tf.math.exp(kl_divergence)
        return inception_score_per_batch

    @staticmethod
    def get_or_train_inception(
        dataset: tf.data.Dataset,
        name: str,
        num_classes: int,
        epochs: int,
        fine_tuning: bool = False,
        loss_fn: tf.keras.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(1e-5),
        logdir: Union[Path, str] = Path().cwd() / "log",
    ) -> tf.keras.Model:
        """
        Restore or train (and save) the Inception model.

        Args:
            dataset (:py:class:`tf.data.Dataset`): Dataset to re-train Inception Model on.
            name (str): Name of this new Inception Model, used for saving it.
            num_classes (int): Number of classes to use for classification.
            epochs (int): Epochs to train the Inception model for.
            fine_tuning (bool): Controls wether the model will be fine-tuned or used as is.
            loss_fn (:py:class:`tf.keras.losses.Loss`): Keras Loss for the model.
            optimizer (:py:class:`tf.keras.optimizers.Optimizer`): Keras optimizer for the model.
            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.

        Returns:
            :py:class:`tf.keras.Model`: The Inception Model.

        """
        os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
        model = tf.keras.Sequential(
            [
                hub.KerasLayer(
                    "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
                    output_shape=[2048],
                    trainable=fine_tuning,
                ),
                tf.keras.layers.Dense(512),
                tf.keras.layers.LeakyReLU(alpha=0.05),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        del os.environ["TFHUB_DOWNLOAD_PROGRESS"]
        step = tf.Variable(0, trainable=False, dtype=tf.int64)

        ckpt = tf.train.Checkpoint()
        ckpt.objects = []
        ckpt.objects.extend([model, step])
        logdir = logdir
        manager = tf.train.CheckpointManager(
            ckpt, logdir / "inception", name, max_to_keep=1
        )

        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            print(f"Restored checkpoint {manager.latest_checkpoint}.")
            return model

        print("Training the InceptionV3 model")

        # callback checkpoint
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(logdir)
        model.compile(loss=loss_fn, optimizer=optimizer)
        model.fit(dataset, epochs=epochs, callbacks=[model_checkpoint_callback])

        return model


class EncodingAccuracy(ClassifierMetric):
    """
    Generator and Encoder accuracy performance.

    Measure the Generator and Encoder performance together, by classifying:
    `G(E(x)), y` using a pre-trained classified (on the dataset of x).

    """

    def __init__(
        self,
        classifier: tf.keras.Model,
        name: str = "encoding_accuracy",
        model_selection_operator: Callable = None,
        logdir=Path().cwd() / "log",
    ) -> None:
        """
        Measure the Generator and Encoder performance together.

        This is done by classifying: `G(E(x)), y` using a pre-trained classified
        (on the dataset of x).

        Args:
            classifier (:py:class:`tf.keras.Model`): Keras Model to use as a Classifier to
                measure the accuracy. Generally assumed to be the Inception Model.
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
            metric=tf.metrics.Accuracy(name),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._classifier = classifier

    def update_state(self, context: GANEncoderContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.GANEncoderContext`): An AshPy Context Object
                that carries all the information the Metric needs.

        """
        inner_context = types.SimpleNamespace()
        inner_context.classifier_model = self._classifier
        inner_context.log_eval_mode = LogEvalMode.TEST

        # Return G(E(x)), y
        def _gen(real_xy, _):
            real_x, real_y = real_xy
            out = context.generator_model(
                context.encoder_model(
                    real_x, training=context.log_eval_mode == LogEvalMode.TRAIN
                ),
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )
            return out, real_y

        dataset = context.dataset.map(_gen)
        inner_context.dataset = dataset
        # Classify using the pre-trained classifier (self._classifier)
        # G(E(x)) and check the accuracy (with y)
        super().update_state(inner_context)
