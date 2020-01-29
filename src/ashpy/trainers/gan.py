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

"""Collection of GANs trainers."""
from pathlib import Path
from typing import List, Optional, Union

import tensorflow as tf
from ashpy.callbacks import Callback
from ashpy.contexts.gan import GANContext, GANEncoderContext
from ashpy.datasets import wrap
from ashpy.losses.executor import Executor
from ashpy.metrics import Metric
from ashpy.metrics.gan import DiscriminatorLoss, EncoderLoss, GeneratorLoss
from ashpy.modes import LogEvalMode
from ashpy.trainers.trainer import Trainer

__ALL__ = ["AdversarialTrainer", "EncoderTrainer"]


class AdversarialTrainer(Trainer):
    r"""
    Primitive Trainer for GANs subclassed from :class:`ashpy.trainers.Trainer`.

    Examples:
        .. testcode::

            import shutil
            import operator

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
            logdir = "testlog/adversarial"
            metrics = [
                metrics.gan.InceptionScore(
                    # Fake inception model
                    models.gans.ConvDiscriminator(
                        layer_spec_input_res=(299, 299),
                        layer_spec_target_res=(7, 7),
                        kernel_size=(5, 5),
                        initial_filters=16,
                        filters_cap=32,
                        output_shape=10,
                    ),
                    model_selection_operator=operator.gt,
                )
            ]
            trainer = trainers.gan.AdversarialTrainer(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=tf.optimizers.Adam(1e-4),
                discriminator_optimizer=tf.optimizers.Adam(1e-4),
                generator_loss=generator_bce,
                discriminator_loss=minmax,
                epochs=epochs,
                metrics=metrics,
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

        .. testoutput::

            Initializing checkpoint.
            Starting epoch 1.
            [1] Saved checkpoint: testlog/adversarial/ckpts/ckpt-1
            Epoch 1 completed.
            Starting epoch 2.
            [2] Saved checkpoint: testlog/adversarial/ckpts/ckpt-2
            Epoch 2 completed.
            Training finished after 2 epochs.

    """

    ckpt_id_generator: str = "generator"
    ckpt_id_discriminator: str = "discriminator"
    ckpt_id_optimizer_generator: str = "optimizer_generator"
    ckpt_id_optimizer_discriminator: str = "optimizer_discriminator"

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        generator_optimizer: tf.optimizers.Optimizer,
        discriminator_optimizer: tf.optimizers.Optimizer,
        generator_loss: Executor,
        discriminator_loss: Executor,
        epochs: int,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
        logdir: Union[Path, str] = Path().cwd() / "log",
        log_eval_mode: LogEvalMode = LogEvalMode.TEST,
        global_step: Optional[tf.Variable] = None,
    ):
        r"""
        Instantiate a :py:class:`AdversarialTrainer`.

        Args:
            generator (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model`
                describing the Generator part of a GAN.
            discriminator (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model`
                describing the Discriminator part of a GAN.
            generator_optimizer (:py:class:`tf.optimizers.Optimizer`): A :py:mod:`tf.optimizers`
                to use for the Generator.
            discriminator_optimizer (:py:class:`tf.optimizers.Optimizer`): A
                :py:mod:`tf.optimizers` to use for the Discriminator.
            generator_loss (:py:class:`ashpy.losses.executor.Executor`): A ash Executor to compute
                the loss of the Generator.
            discriminator_loss (:py:class:`ashpy.losses.executor.Executor`): A ash Executor
                to compute the loss of the Discriminator.
            epochs (int): number of training epochs.
            metrics: (List): list of :py:class:`ashpy.metrics.Metric` to measure on
                training and validation data.
            callbacks (List): list of :py:class:`ashpy.callbacks.Callback` to measure on
                training and validation data.
            logdir: checkpoint and log directory.
            log_eval_mode: models' mode to use when evaluating and logging.
            global_step (Optional[:py:class:`tf.Variable`]): tf.Variable that
                keeps track of the training steps.

        Returns:
            :py:obj:`None`

        """
        super().__init__(
            epochs=epochs,
            logdir=logdir,
            log_eval_mode=log_eval_mode,
            global_step=global_step,
            callbacks=callbacks,
            example_dim=(2, 1),
        )
        self._generator = generator
        self._discriminator = discriminator

        self._generator_loss = generator_loss
        self._generator_loss.reduction = tf.losses.Reduction.NONE

        self._discriminator_loss = discriminator_loss
        self._discriminator_loss.reduction = tf.losses.Reduction.NONE

        losses_metrics = [
            DiscriminatorLoss(name="ashpy/d_loss", logdir=logdir),
            GeneratorLoss(name="ashpy/g_loss", logdir=logdir),
        ]
        if metrics:
            metrics.extend(losses_metrics)
        else:
            metrics = losses_metrics

        super()._update_metrics(metrics)
        super()._validate_metrics()

        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer

        ckpt_dict = {
            self.ckpt_id_optimizer_generator: self._generator_optimizer,
            self.ckpt_id_optimizer_discriminator: self._discriminator_optimizer,
            self.ckpt_id_generator: self._generator,
            self.ckpt_id_discriminator: self._discriminator,
        }
        self._update_checkpoint(ckpt_dict)

        # pylint: disable=unidiomatic-typecheck
        if type(self) == AdversarialTrainer:
            self._restore_or_init()

        self._context = GANContext(
            generator_model=self._generator,
            discriminator_model=self._discriminator,
            generator_loss=self._generator_loss,
            discriminator_loss=self._discriminator_loss,
            log_eval_mode=self._log_eval_mode,
            global_step=self._global_step,
            checkpoint=self._checkpoint,
            metrics=self._metrics,
        )

    def train_step(self, real_xy, g_inputs):
        """
        Train step for the AdversarialTrainer.

        Args:
            real_xy: input batch as extracted from the input dataset.
                     (features, label) pair.
            g_inputs: batch of generator_input as generated from the input dataset.

        Returns:
            d_loss, g_loss, fake: discriminator, generator loss values. fake is the
                generator output.

        """
        real_x, real_y = real_xy

        if len(self._generator.inputs) == 2:
            g_inputs = [g_inputs, real_y]

        with tf.GradientTape(persistent=True) as tape:
            fake = self._generator(g_inputs, training=True)

            d_loss = self._discriminator_loss(
                self._context, fake=fake, real=real_x, condition=real_y, training=True
            )

            g_loss = self._generator_loss(
                self._context, fake=fake, real=real_x, condition=real_y, training=True
            )

        # check that we have some trainable_variables
        assert self._generator.trainable_variables
        assert self._discriminator.trainable_variables

        # calculate the gradient
        d_gradients = tape.gradient(d_loss, self._discriminator.trainable_variables)
        g_gradients = tape.gradient(g_loss, self._generator.trainable_variables)

        # delete the tape since it's persistent
        del tape

        # apply the gradient
        self._discriminator_optimizer.apply_gradients(
            zip(d_gradients, self._discriminator.trainable_variables)
        )
        self._generator_optimizer.apply_gradients(
            zip(g_gradients, self._generator.trainable_variables)
        )

        return d_loss, g_loss, fake

    @tf.function
    def _train_step(self, example):
        """Training step with the distribution strategy."""
        ret = self._distribute_strategy.experimental_run_v2(
            self.train_step, args=(example[0], example[1])
        )

        per_replica_losses = ret[:-1]
        fake = ret[-1]

        return (
            self._reduce(per_replica_losses[0], tf.distribute.ReduceOp.SUM),
            self._reduce(per_replica_losses[1], tf.distribute.ReduceOp.SUM),
            fake,
        )

    def call(
        self,
        dataset: tf.data.Dataset,
        log_freq: int = 10,
        measure_performance_freq: int = 10,
    ):
        """
        Perform the adversarial training.

        Args:
            dataset (:py:obj:`tf.data.Dataset`): The adversarial training dataset.
            log_freq (int): Specifies how many steps to run before logging the losses,
                e.g. `log_frequency=10` logs every 10 steps of training.
                Pass `log_frequency<=0` in case you don't want to log.
            measure_performance_freq (int): Specifies how many steps to run before
                measuring the performance, e.g. `measure_performance_freq=10`
                measures performance every 10 steps of training.
                Pass `measure_performance_freq<=0` in case you don't want to measure
                    performance.

        """
        current_epoch = self._current_epoch()

        self._update_global_batch_size(
            dataset, [self._discriminator_loss, self._generator_loss]
        )

        dataset = wrap(
            dataset.unbatch().batch(self._global_batch_size, drop_remainder=True)
        )
        samples = next(iter(dataset.take(1)))

        self._context.generator_inputs = samples[1]

        with self._train_summary_writer.as_default():

            # notify on train start
            self._on_train_start()

            for _ in tf.range(current_epoch, self._epochs):
                distribute_dataset = self._distribute_strategy.experimental_distribute_dataset(
                    dataset
                )

                # notify on epoch start
                self._on_epoch_start()

                for example in distribute_dataset:

                    # notify on batch start
                    self._on_batch_start()

                    # perform training step
                    d_loss, g_loss, fake = self._train_step(example)

                    # store fake samples in the context
                    self._context.fake_samples = fake

                    self._global_step.assign_add(1)

                    # print statistics
                    if log_freq > 0 and tf.equal(
                        tf.math.mod(self._global_step, log_freq), 0
                    ):
                        tf.print(
                            f"[{self._global_step.numpy()}] g_loss: {g_loss} - d_loss: {d_loss}"
                        )

                    # measure performance if needed
                    self._measure_performance_if_needed(
                        example, measure_performance_freq
                    )

                    # notify on batch end
                    self._on_batch_end()

                # notify on epoch end
                self._on_epoch_end()

            # final callback
            self._on_train_end()


class EncoderTrainer(AdversarialTrainer):
    r"""
    Primitive Trainer for GANs using an Encoder sub-network.

    The implementation is thought to be used with the BCE losses. To use another loss function
    consider subclassing the model and overriding the train_step method.

    Examples:
        .. testcode::

            from pathlib import Path

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

            # Fake pre-trained classifier
            num_classes = 1
            classifier = tf.keras.Sequential(
                [tf.keras.layers.Dense(10), tf.keras.layers.Dense(num_classes)]
            )

            logdir = Path("testlog") / "adversarial_encoder"

            if logdir.exists():
                shutil.rmtree(logdir)

            metrics = [metrics.gan.EncodingAccuracy(classifier)]

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
                metrics=metrics,
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
            [10] Saved checkpoint: testlog/adversarial_encoder/ckpts/ckpt-1
            Epoch 1 completed.
            Starting epoch 2.
            [20] Saved checkpoint: testlog/adversarial_encoder/ckpts/ckpt-2
            Epoch 2 completed.
            Training finished after 2 epochs.

    """
    ckpt_id_encoder: str = "encoder"
    ckpt_id_optimizer_encoder: str = "optimizer_encoder"

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        encoder: tf.keras.Model,
        generator_optimizer: tf.optimizers.Optimizer,
        discriminator_optimizer: tf.optimizers.Optimizer,
        encoder_optimizer: tf.optimizers.Optimizer,
        generator_loss: Executor,
        discriminator_loss: Executor,
        encoder_loss: Executor,
        epochs: int,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
        logdir: Union[Path, str] = Path().cwd() / "log",
        log_eval_mode: LogEvalMode = LogEvalMode.TEST,
        global_step: Optional[tf.Variable] = None,
    ):
        r"""
        Instantiate a :py:class:`EncoderTrainer`.

        Args:
            generator (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model`
                describing the Generator part of a GAN.
            discriminator (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model`
                describing the Discriminator part of a GAN.
            encoder (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model` describing
                the Encoder part of a GAN.
            generator_optimizer (:py:class:`tf.optimizers.Optimizer`): A :py:mod:`tf.optimizers`
                to use for the Generator.
            discriminator_optimizer (:py:class:`tf.optimizers.Optimizer`): A :py:mod:`tf.optimizers`
                to use for the Discriminator.
            encoder_optimizer (:py:class:`tf.optimizers.Optimizer`): A :py:mod:`tf.optimizers`
                to use for the Encoder.
            generator_loss (:py:class:`ashpy.losses.executor.Executor`): A ash Executor to compute
                the loss of the Generator.
            discriminator_loss (:py:class:`ashpy.losses.executor.Executor`): A ash Executor to
                compute the loss of the Discriminator.
            encoder_loss (:py:class:`ashpy.losses.executor.Executor`): A ash Executor to compute
                the loss of the Discriminator.
            epochs (int): number of training epochs.
            metrics: (List): list of ashpy.metrics.Metric to measure on training and
                validation data.
            callbacks (List): List of ashpy.callbacks.Callback to call on events
            logdir: checkpoint and log directory.
            log_eval_mode (:py:class:`ashpy.modes.LogEvalMode`): models' mode to use
                when evaluating and logging.
            global_step (Optional[:py:class:`tf.Variable`]): tf.Variable that keeps
                track of the training steps.

        """
        if not metrics:
            metrics = []
        metrics.append(EncoderLoss(name="ashpy/e_loss", logdir=logdir))
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            epochs=epochs,
            metrics=metrics,
            callbacks=callbacks,
            logdir=logdir,
            log_eval_mode=log_eval_mode,
            global_step=global_step,
        )

        self._encoder = encoder
        self._encoder_optimizer = encoder_optimizer

        self._encoder_loss = encoder_loss
        self._encoder_loss.reduction = tf.losses.Reduction.NONE

        ckpt_dict = {
            self.ckpt_id_encoder: self._encoder,
            self.ckpt_id_optimizer_encoder: self._encoder_optimizer,
        }
        self._update_checkpoint(ckpt_dict)

        self._restore_or_init()

        self._context = GANEncoderContext(
            generator_model=self._generator,
            discriminator_model=self._discriminator,
            encoder_model=self._encoder,
            generator_loss=self._generator_loss,
            discriminator_loss=self._discriminator_loss,
            encoder_loss=self._encoder_loss,
            log_eval_mode=self._log_eval_mode,
            global_step=self._global_step,
            checkpoint=self._checkpoint,
            metrics=self._metrics,
        )

    def train_step(self, real_xy, g_inputs):
        """Adversarial training step.

        Args:
            real_xy: input batch as extracted from the discriminator input dataset.
                     (features, label) pair
            g_inputs: batch of noise as generated by the generator input dataset.

        Returns:
            d_loss, g_loss, e_loss: discriminator, generator, encoder loss values.

        """
        real_x, real_y = real_xy

        if len(self._generator.inputs) == 2:
            g_inputs = [g_inputs, real_y]

        with tf.GradientTape(persistent=True) as tape:
            fake = self._generator(g_inputs, training=True)

            g_loss = self._generator_loss(
                self._context, fake=fake, real=real_x, condition=real_y, training=True
            )

            d_loss = self._discriminator_loss(
                self._context, fake=fake, real=real_x, condition=real_y, training=True
            )

            e_loss = self._encoder_loss(
                self._context, fake=fake, real=real_x, condition=real_y, training=True
            )

        g_gradients = tape.gradient(g_loss, self._generator.trainable_variables)
        d_gradients = tape.gradient(d_loss, self._discriminator.trainable_variables)
        e_gradients = tape.gradient(e_loss, self._encoder.trainable_variables)
        del tape

        # Only for logging in special cases (out of tape)
        generator_of_encoder = self._generator(
            self._encoder(real_x, training=True), training=True
        )

        self._discriminator_optimizer.apply_gradients(
            zip(d_gradients, self._discriminator.trainable_variables)
        )
        self._generator_optimizer.apply_gradients(
            zip(g_gradients, self._generator.trainable_variables)
        )
        self._encoder_optimizer.apply_gradients(
            zip(e_gradients, self._encoder.trainable_variables)
        )

        return d_loss, g_loss, e_loss, fake, generator_of_encoder

    @tf.function
    def _train_step(self, example):
        """Perform the training step using the distribution strategy."""
        ret = self._distribute_strategy.experimental_run_v2(
            self.train_step, args=(example[0], example[1])
        )

        per_replica_losses = ret[:3]
        fake = ret[3]
        generator_of_encoder = ret[4]
        return (
            self._reduce(per_replica_losses[0], tf.distribute.ReduceOp.SUM),
            self._reduce(per_replica_losses[1], tf.distribute.ReduceOp.SUM),
            self._reduce(per_replica_losses[2], tf.distribute.ReduceOp.SUM),
            fake,
            generator_of_encoder,
        )

    def call(
        self,
        dataset: tf.data.Dataset,
        log_freq: int = 10,
        measure_performance_freq: int = 10,
    ):
        r"""
        Perform the adversarial training.

        Args:
            dataset (:py:class:`tf.data.Dataset`): The adversarial training dataset.
            log_freq (int): Specifies how many steps to run before logging the losses,
                e.g. `log_frequency=10` logs every 10 steps of training.
                Pass `log_frequency<=0` in case you don't want to log.
            measure_performance_freq (int): Specifies how many steps to run before
                measuring the performance, e.g. `measure_performance_freq=10`
                measures performance every 10 steps of training.
                Pass `measure_performance_freq<=0` in case you don't want to measure
                performance.

        """
        current_epoch = self._current_epoch()

        self._update_global_batch_size(
            dataset,
            [self._discriminator_loss, self._generator_loss, self._encoder_loss],
        )

        dataset = wrap(
            dataset.unbatch().batch(self._global_batch_size, drop_remainder=True)
        )

        samples = next(iter(dataset.take(1)))

        self._context.generator_inputs = samples[1]
        self._context.encoder_inputs = samples[0][0]

        with self._train_summary_writer.as_default():

            # notify on train start event
            self._on_train_start()

            for _ in tf.range(current_epoch, self._epochs):

                distribute_dataset = self._distribute_strategy.experimental_distribute_dataset(
                    dataset
                )

                # notify on epoch start event
                self._on_epoch_start()

                for example in distribute_dataset:

                    # perform training step
                    (
                        d_loss,
                        g_loss,
                        e_loss,
                        fake,
                        generator_of_encoder,
                    ) = self._train_step(example)

                    # increase global step
                    self._global_step.assign_add(1)

                    # setup fake_samples
                    self._context.fake_samples = fake
                    self._context.generator_of_encoder = generator_of_encoder

                    # Log losses
                    if log_freq > 0 and tf.equal(
                        tf.math.mod(self._global_step, log_freq), 0
                    ):
                        tf.print(
                            f"[{self._global_step.numpy()}] g_loss: {g_loss} - "
                            f"d_loss: {d_loss} - e_loss: {e_loss}"
                        )

                    # measure performance if needed
                    self._measure_performance_if_needed(
                        example, measure_performance_freq
                    )

                    # notify on batch end event
                    self._on_batch_end()

                # notify on epoch end event
                self._on_epoch_end()

            # notify on training end event
            self._on_train_end()
