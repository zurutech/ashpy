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

"""
GAN callbacks.

LogImageGANCallback: Log output of the generator when evaluated in its inputs.
LogImageGANEncoderCallback: Log output of the generator when evaluated in the encoder
"""

import tensorflow as tf
from ashpy import LogEvalMode
from ashpy.callbacks.counter_callback import CounterCallback
from ashpy.callbacks.events import Event
from ashpy.contexts import GANContext, GANEncoderContext
from ashpy.utils.utils import log


class LogImageGANCallback(CounterCallback):
    """
    Callback used for logging GANs images to Tensorboard.

    Logs the Generator output.
    Logs G(z).

    Examples:
        .. testcode::

            import shutil
            import operator
            from pathlib import Path

            generator = models.gans.ConvGenerator(
                layer_spec_input_res=(7, 7),
                layer_spec_target_res=(28, 28),
                kernel_size=(5, 5),
                initial_filters=32,
                filters_cap=16,
                channels=1,
            )

            discriminator = models.gans.ConvDiscriminator(
                layer_spec_input_res=(28, 28),
                layer_spec_target_res=(7, 7),
                kernel_size=(5, 5),
                initial_filters=16,
                filters_cap=32,
                output_shape=1,
            )

            # Losses
            generator_bce = losses.gan.GeneratorBCE()
            minmax = losses.gan.DiscriminatorMinMax()

            # Real data
            batch_size = 2
            mnist_x, mnist_y = tf.zeros((100,28,28)), tf.zeros((100,))

            # Trainer
            epochs = 2
            logdir = Path("testlog/callbacks")
            callbacks = [callbacks.LogImageGANCallback()]
            trainer = trainers.gan.AdversarialTrainer(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=tf.optimizers.Adam(1e-4),
                discriminator_optimizer=tf.optimizers.Adam(1e-4),
                generator_loss=generator_bce,
                discriminator_loss=minmax,
                epochs=epochs,
                callbacks=callbacks,
                logdir=logdir,
            )

            # take only 2 samples to speed up tests
            real_data = (
                tf.data.Dataset.from_tensor_slices(
                (tf.expand_dims(mnist_x, -1), tf.expand_dims(mnist_y, -1))).take(batch_size)
                .batch(batch_size)
                .prefetch(1)
            )

            # Add noise in the same dataset, just by mapping.
            # The return type of the dataset must be: tuple(tuple(a,b), noise)
            dataset = real_data.map(
                lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, 100)))
            )

            trainer(dataset)
            shutil.rmtree(logdir)

            assert not logdir.exists()

            trainer._global_step.assign_add(500)

        .. testoutput::

            Initializing checkpoint.
            Starting epoch 1.
            [1] Saved checkpoint: testlog/callbacks/ckpts/ckpt-1
            Epoch 1 completed.
            Starting epoch 2.
            [2] Saved checkpoint: testlog/callbacks/ckpts/ckpt-2
            Epoch 2 completed.
            Training finished after 2 epochs.

    """

    def __init__(
        self,
        event: Event = Event.ON_EPOCH_END,
        name: str = "log_image_gan_callback",
        event_freq: int = 1,
    ) -> None:
        """
        Initialize the LogImageCallbackGAN.

        Args:
            event (:py:class:`ashpy.callbacks.events.Event`): event to consider.
            event_freq (int): frequency of logging.
            name (str): name of the callback.

        """
        super(LogImageGANCallback, self).__init__(
            event=event, fn=self._log_fn, name=name, event_freq=event_freq
        )

    def _log_fn(self, context: GANContext) -> None:
        """
        Log output of the generator to Tensorboard.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANContext`): current context.

        """
        if context.log_eval_mode == LogEvalMode.TEST:
            out = context.generator_model(context.generator_inputs, training=False)
        elif context.log_eval_mode == LogEvalMode.TRAIN:
            out = context.fake_samples
        else:
            raise ValueError("Invalid LogEvalMode")

        # tensorboard 2.0 does not support float images in [-1, 1]
        # only in [0,1]
        if out.dtype == tf.float32:
            # The hypothesis is that image are in [-1,1] how to check?
            out = (out + 1.0) / 2

        log("generator", out, context.global_step)


class LogImageGANEncoderCallback(LogImageGANCallback):
    """
    Callback used for logging GANs images to Tensorboard.

    Logs the Generator output evaluated in the encoder output.
    Logs G(E(x)).

    Examples:
        .. testcode::

            import shutil
            import operator

            def real_gen():
                label = 0
                for _ in tf.range(100):
                    yield ((10.0,), (label,))

            latent_dim = 100

            generator = tf.keras.Sequential([tf.keras.layers.Dense(1)])

            left_input = tf.keras.layers.Input(shape=(1,))
            left = tf.keras.layers.Dense(10, activation=tf.nn.elu)(left_input)

            right_input = tf.keras.layers.Input(shape=(latent_dim,))
            right = tf.keras.layers.Dense(10, activation=tf.nn.elu)(right_input)

            net = tf.keras.layers.Concatenate()([left, right])
            out = tf.keras.layers.Dense(1)(net)

            discriminator = tf.keras.Model(inputs=[left_input, right_input], outputs=[out])

            encoder = tf.keras.Sequential([tf.keras.layers.Dense(latent_dim)])

            # Losses
            generator_bce = losses.gan.GeneratorBCE()
            encoder_bce = losses.gan.EncoderBCE()
            minmax = losses.gan.DiscriminatorMinMax()

            epochs = 2

            callbacks = [callbacks.LogImageGANEncoderCallback()]

            logdir = "testlog/callbacks_encoder"

            trainer = trainers.gan.EncoderTrainer(
                generator=generator,
                discriminator=discriminator,
                encoder=encoder,
                discriminator_optimizer=tf.optimizers.Adam(1e-4),
                generator_optimizer=tf.optimizers.Adam(1e-5),
                encoder_optimizer=tf.optimizers.Adam(1e-6),
                generator_loss=generator_bce,
                discriminator_loss=minmax,
                encoder_loss=encoder_bce,
                epochs=epochs,
                callbacks=callbacks,
                logdir=logdir,
            )

            batch_size = 10
            discriminator_input = tf.data.Dataset.from_generator(
                real_gen, (tf.float32, tf.int64), ((1), (1))
            ).batch(batch_size)

            dataset = discriminator_input.map(
                lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, latent_dim)))
            )

            trainer(dataset)

            shutil.rmtree(logdir)

        .. testoutput::

            Initializing checkpoint.
            Starting epoch 1.
            [10] Saved checkpoint: testlog/callbacks_encoder/ckpts/ckpt-1
            Epoch 1 completed.
            Starting epoch 2.
            [20] Saved checkpoint: testlog/callbacks_encoder/ckpts/ckpt-2
            Epoch 2 completed.
            Training finished after 2 epochs.

    """

    def __init__(
        self,
        event: Event = Event.ON_EPOCH_END,
        name: str = "log_image_gan_encoder_callback",
        event_freq: int = 1,
    ) -> None:
        """
        Initialize the LogImageGANEncoderCallback.

        Args:
            event (:py:class:`ashpy.callbacks.events.Event`): event to consider.
            event_freq (int): frequency of logging.
            name (str): name of the callback.

        """
        super(LogImageGANEncoderCallback, self).__init__(
            event=event, name=name, event_freq=event_freq
        )

    def _log_fn(self, context: GANEncoderContext):
        """
        Log output of the generator to Tensorboard.

        Logs G(E(x)).

        Args:
            context (:py:class:`ashpy.contexts.gan.GanEncoderContext`): current context.

        """
        if context.log_eval_mode == LogEvalMode.TEST:
            generator_of_encoder = context.generator_model(
                context.encoder_model(context.encoder_inputs, training=False),
                training=False,
            )
        elif context.log_eval_mode == LogEvalMode.TRAIN:
            generator_of_encoder = context.generator_of_encoder
        else:
            raise ValueError("Invalid LogEvalMode")

        # Tensorboard 2.0 does not support float images in [-1, 1]
        # Only in [0,1]
        if generator_of_encoder.dtype == tf.float32:
            # The hypothesis is that image are in [-1,1] how to check?
            generator_of_encoder = (generator_of_encoder + 1.0) / 2

        log("generator_of_encoder", generator_of_encoder, context.global_step)
