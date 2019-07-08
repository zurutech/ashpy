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
import tensorflow as tf

from ashpy.contexts.base_context import BaseContext
from ashpy.modes import LogEvalMode


class GANContext(BaseContext):
    """:py:class:`ashpy.contexts.gan.GANContext` measure the specified metrics on the GAN."""

    def __init__(
        self,
        dataset=None,
        generator_model=None,
        discriminator_model=None,
        generator_loss=None,
        discriminator_loss=None,
        metrics=None,
        log_eval_mode=LogEvalMode.TRAIN,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        ckpt=None,
    ):
        """
        Initialize the (:py:class:`ashpy.contexts.gan.GANContext`).

        Args:
            dataset (:py:class:`tf.data.Dataset`): Dataset of tuples. [0] true dataset, [1] generator input dataset
            generator_model: the generator.
            discriminator_model: the discriminator.
            generator_loss: the generator loss.
            discriminator_loss: the discriminator loss.
            metrics: All the metrics to be used to evaluate the model.
            log_eval_mode: models' mode to  use when evaluating and logging.
            global_step: tf.Variable that keeps track of the training steps.
            ckpt (:py:class:`tf.train.Checkpoint`): checkpoint to use to keep track of models status.
        """
        super().__init__(metrics, dataset, log_eval_mode, global_step, ckpt)

        self._generator_model = generator_model
        self._discriminator_model = discriminator_model

        self._generator_loss = generator_loss
        self._discriminator_loss = discriminator_loss

    @property
    def generator_model(self):
        """Returns the generator model"""
        return self._generator_model

    @property
    def discriminator_model(self):
        """Returns the discriminator model"""
        return self._discriminator_model

    @property
    def generator_loss(self):
        """Returns the generator loss"""
        return self._generator_loss

    @property
    def discriminator_loss(self):
        """Returns the discriminator loss"""
        return self._discriminator_loss


class GANEncoderContext(GANContext):
    r""":py:class:`ashpy.contexts.gan.GANEncoderContext` measure the specified metrics on the GAN."""

    def __init__(
        self,
        dataset=None,
        generator_model=None,
        discriminator_model=None,
        encoder_model=None,
        generator_loss=None,
        discriminator_loss=None,
        encoder_loss=None,
        metrics=None,
        log_eval_mode=LogEvalMode.TRAIN,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        ckpt=None,
    ):
        r"""
        Initialize the :py:class:`ashpy.contexts.gan.GANEncoderContext`.

        Args:
            dataset (:py:class:`tf.data.Dataset`): Dataset of tuples. [0] true dataset, [1] generator input dataset
            generator_model: the generator.
            discriminator_model: the discriminator.
            encoder_model: the encoder.
            generator_loss: the generator loss.
            discriminator_loss: the discriminator loss.
            encoder_loss: the encoder loss.
            metrics: All the metrics to be used to evaluate the model.
            log_eval_mode: models' mode to  use when evaluating and logging.
            global_step: tf.Variable that keeps track of the training steps.
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
    def encoder_model(self):
        """Returns the encoder model"""
        return self._encoder_model

    @property
    def encoder_loss(self):
        """Returns the encoder loss"""
        return self._encoder_loss
