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
from ashpy.contexts.context import Context
from ashpy.modes import LogEvalMode

if TYPE_CHECKING:
    from ashpy.losses.executor import Executor
    from ashpy.metrics import Metric


class GANContext(Context):
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
        checkpoint: tf.train.Checkpoint = None,
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
            checkpoint (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of
                models status.

        """
        super().__init__(metrics, dataset, log_eval_mode, global_step, checkpoint)

        self._generator_model = generator_model
        self._discriminator_model = discriminator_model

        self._generator_loss = generator_loss
        self._discriminator_loss = discriminator_loss

        self._fake_samples = None
        self._generator_inputs = None

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

    @property
    def fake_samples(self) -> Optional[tf.Tensor]:
        """Retrieve the fake samples, i.e. output of the generator."""
        return self._fake_samples

    @fake_samples.setter
    def fake_samples(self, _fake_samples: Optional[tf.Tensor]):
        """Set the fake samples, i.e. output of the generator."""
        self._fake_samples = _fake_samples

    @property
    def generator_inputs(self) -> Optional[tf.Tensor]:
        """Retrieve the generator inputs."""
        return self._generator_inputs

    @generator_inputs.setter
    def generator_inputs(self, _generator_inputs: Optional[tf.Tensor]):
        """Set the generator inputs."""
        self._generator_inputs = _generator_inputs


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
        checkpoint: tf.train.Checkpoint = None,
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
            checkpoint (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of
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
            checkpoint=checkpoint,
        )
        self._encoder_model = encoder_model
        self._encoder_loss = encoder_loss

        self._generator_of_encoder = None
        self._encoder_inputs = None

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

    @property
    def generator_of_encoder(self) -> tf.Tensor:
        """Retrieve the images generated from the encoder output."""
        return self._generator_of_encoder

    @generator_of_encoder.setter
    def generator_of_encoder(self, _generator_of_encoder: tf.Tensor):
        """Set the images generated from the encoder output."""
        self._generator_of_encoder = _generator_of_encoder

    @property
    def encoder_inputs(self) -> tf.Tensor:
        """Retrieve the inputs of the encoder."""
        return self._encoder_inputs

    @encoder_inputs.setter
    def encoder_inputs(self, _encoder_inputs: tf.Tensor):
        """Setter for the inputs of the encoder."""
        self._encoder_inputs = _encoder_inputs
