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

"""GANContext measures the specified metrics on the GAN."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import tensorflow as tf

from ashpy.contexts.base_context import BaseContext
from ashpy.modes import LogEvalMode

if TYPE_CHECKING:
    from ashpy.losses.executor import Executor
    from ashpy.metrics import Metric


class GANContext(BaseContext):
    """:py:class:`ashpy.contexts.gan.GANContext` measure the specified metrics on the GAN."""

    def __init__(
        self,
        dataset: tf.data.Dataset = None,
        generator_model: tf.keras.Model = None,
        discriminator_model: tf.keras.Model = None,
        generator_loss: Executor = None,
        discriminator_loss: Executor = None,
        metrics: List[Metric] = None,
        log_eval_mode: LogEvalMode = LogEvalMode.TRAIN,
        global_step: tf.Variable = tf.Variable(
            0, name="global_step", trainable=False, dtype=tf.int64
        ),
        ckpt: tf.train.Checkpoint = None,
    ) -> None:
        """
        Initialize the Context.

        Args:
            dataset (:py:class:`tf.data.Dataset`): Dataset of tuples. [0] true dataset,
                [1] generator input dataset.
            generator_model (:py:class:`tf.keras.Model`): The generator.
            discriminator_model (:py:class:`tf.keras.Model`): The discriminator.
            generator_loss (:py:func:`ashpy.losses.Executor`): The generator loss.
            discriminator_loss (:py:func:`ashpy.losses.Executor`): The discriminator loss.
            metrics (:obj:`list` of [:py:class:`ashpy.metrics.metric.Metric`]): All the metrics
                to be used to evaluate the model.
            log_eval_mode (:py:class:`ashpy.modes.LogEvalMode`): Models' mode to  use when
                evaluating and logging.
            global_step (:py:class:`tf.Variable`): `tf.Variable` that keeps track of the
                training steps.
            ckpt (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of
                models status.

        """
        super().__init__(metrics, dataset, log_eval_mode, global_step, ckpt)

        self._generator_model = generator_model
        self._discriminator_model = discriminator_model

        self._generator_loss = generator_loss
        self._discriminator_loss = discriminator_loss

    @property
    def generator_model(self) -> tf.keras.Model:
        """
        Retrieve the generator model.

        Returns:
            :py:class:`tf.keras.Model`.

        """
        return self._generator_model

    @property
    def discriminator_model(self) -> tf.keras.Model:
        """
        Retrieve the discriminator model.

        Returns:
            :py:class:`tf.keras.Model`.

        """
        return self._discriminator_model

    @property
    def generator_loss(self) -> Optional[Executor]:
        """Retrieve the generator loss."""
        return self._generator_loss

    @property
    def discriminator_loss(self) -> Optional[Executor]:
        """Retrieve the discriminator loss."""
        return self._discriminator_loss


class GANEncoderContext(GANContext):
    """:py:class:`ashpy.contexts.gan.GANEncoderContext` measure the specified metrics on the GAN."""

    def __init__(
        self,
        dataset: tf.data.Dataset = None,
        generator_model: tf.keras.Model = None,
        discriminator_model: tf.keras.Model = None,
        encoder_model: tf.keras.Model = None,
        generator_loss: Executor = None,
        discriminator_loss: Executor = None,
        encoder_loss: Executor = None,
        metrics: List[Metric] = None,
        log_eval_mode: LogEvalMode = LogEvalMode.TRAIN,
        global_step: tf.Variable = tf.Variable(
            0, name="global_step", trainable=False, dtype=tf.int64
        ),
        ckpt: tf.train.Checkpoint = None,
    ) -> None:
        r"""
        Initialize the Context.

        Args:
            dataset (:py:class:`tf.data.Dataset`): Dataset of tuples. [0] true dataset,
                [1] generator input dataset.
            generator_model (:py:class:`tf.keras.Model`): The generator.
            discriminator_model (:py:class:`tf.keras.Model`): The discriminator.
            encoder_model (:py:class:`tf.keras.Model`): The encoder.
            generator_loss (:py:func:`ashpy.losses.Executor`): The generator loss.
            discriminator_loss (:py:func:`ashpy.losses.Executor`): The discriminator loss.
            encoder_loss (:py:func:`ashpy.losses.Executor`): The encoder loss.
            metrics (:obj:`list` of [:py:class:`ashpy.metrics.metric.Metric`]): All the metrics
                to be used to evaluate the model.
            log_eval_mode (:py:class:`ashpy.modes.LogEvalMode`): Models' mode to  use when
                evaluating and logging.
            global_step (:py:class:`tf.Variable`): `tf.Variable` that keeps track of the
                training steps.
            ckpt (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of
                models status.

        """
        super().__init__(
            dataset=dataset,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            metrics=metrics,
            log_eval_mode=log_eval_mode,
            global_step=global_step,
            ckpt=ckpt,
        )
        self._encoder_model = encoder_model
        self._encoder_loss = encoder_loss

    @property
    def encoder_model(self) -> tf.keras.Model:
        """
        Retrieve the encoder model.

        Returns:
            :py:class:`tf.keras.Model`.

        """
        return self._encoder_model

    @property
    def encoder_loss(self) -> Optional[Executor]:
        """Retrieve the encoder loss."""
        return self._encoder_loss
