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

"""Primitive Trainer Interface."""

from pathlib import Path
from typing import List, Optional, Union

import ashpy
import tensorflow as tf
from ashpy.callbacks import Callback
from ashpy.contexts.classifier import ClassifierContext
from ashpy.datasets import wrap
from ashpy.metrics import Metric
from ashpy.metrics.classifier import ClassifierLoss
from ashpy.trainers.trainer import Trainer

__ALL__ = ["ClassifierTrainer"]


class ClassifierTrainer(Trainer):
    r""":py:class:`ClassifierTrainer` provide the standard training loop for a classifier."""

    ckpt_id_model: str = "model"
    ckpt_id_optimizer: str = "optimizer"

    def __init__(
        self,
        model: tf.keras.models.Model,
        optimizer: tf.optimizers.Optimizer,
        loss: ashpy.losses.ClassifierLoss,
        epochs: int,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
        logdir: Union[Path, str] = Path().cwd() / "log",
        global_step: Optional[tf.Variable] = None,
    ):
        r"""
        Instantiate the :py:class:`ClassifierTrainer` trainer.

        Args:
            model (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model` model.
            optimizer (:py:class:`tf.optimizers.Optimizer`): A
                :py:class:`tf.optimizers.Optimizer`.
            loss (:obj:`ashpy.losses.classifier.ClassifierLoss`): A loss function built following
                :py:mod:`ashpy.executors``.
            epochs (int): Number of training epochs.
            metrics: (List): List of :py:class:`ashpy.metrics.metric.Metric` to
                measure on training and validation data.
            callbacks (List): List of :py:class:`ashpy.callbacks.callback.Callback` to
                to call on events
            logdir (str): Checkpoint and log directory.
            global_step (Optional[py:class:`tf.Variable`]): tf.Variable that keeps
                track of the training steps.

        Examples:
            .. testcode::
                import operator
                import shutil
                import pathlib
                from ashpy.metrics import ClassifierMetric
                from ashpy.trainers.classifier import ClassifierTrainer
                from ashpy.losses.classifier import ClassifierLoss

                def toy_dataset():
                    inputs = tf.expand_dims(tf.range(1, 1000.0), -1)
                    labels = tf.expand_dims(
                        [1 if tf.equal(tf.math.mod(tf.squeeze(i), 2), 0) else 0 for i in inputs], -1
                    )
                    return tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(10).batch(2)


                model = tf.keras.Sequential(
                    [tf.keras.layers.Dense(10, activation=tf.nn.sigmoid), tf.keras.layers.Dense(2)]
                )
                optimizer = tf.optimizers.Adam(1e-3)

                loss = ClassifierLoss(tf.losses.SparseCategoricalCrossentropy(from_logits=True))
                logdir = "testlog"
                epochs = 2

                if pathlib.Path(logdir).exists():
                    shutil.rmtree(logdir)

                metrics = [
                    ClassifierMetric(tf.metrics.Accuracy()),
                    ClassifierMetric(tf.metrics.BinaryAccuracy()),
                ]

                trainer = ClassifierTrainer(model=model,
                                            optimizer=optimizer,
                                            loss=loss,
                                            epochs=epochs,
                                            metrics=metrics,
                                            logdir=logdir)
                train, validation = toy_dataset(), toy_dataset()
                trainer(train, validation)

                shutil.rmtree(logdir)

            .. testoutput::

                Initializing checkpoint.
                Starting epoch 1.
                [500] Saved checkpoint: testlog/ckpts/ckpt-1
                Epoch 1 completed.
                Starting epoch 2.
                [1000] Saved checkpoint: testlog/ckpts/ckpt-2
                Epoch 2 completed.
                Training finished after 2 epochs.

        """
        super().__init__(
            epochs=epochs,
            logdir=logdir,
            global_step=global_step,
            callbacks=callbacks,
            example_dim=(1, 1),
        )

        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._loss.reduction = tf.keras.losses.Reduction.NONE

        self._avg_loss = ClassifierLoss(name="ashpy/avg_loss")
        if metrics:
            metrics.append(self._avg_loss)
        else:
            metrics = [self._avg_loss]

        super()._update_metrics(metrics)
        super()._validate_metrics()

        ckpt_dict = {
            self.ckpt_id_optimizer: self._optimizer,
            self.ckpt_id_model: self._model,
        }
        self._update_checkpoint(ckpt_dict)

        self._restore_or_init()

        self._context = ClassifierContext(
            classifier_model=self._model,
            loss=self._loss,
            metrics=self._metrics,
            log_eval_mode=self._log_eval_mode,
            global_step=self._global_step,
            checkpoint=self._checkpoint,
        )

    def train_step(self, features, labels):
        """
        Train step.

        Args:
            features: Input features.
            labels: The labels.

        Returns:
            Loss value.

        """
        with tf.GradientTape() as tape:
            loss = self._loss(
                self._context, features=features, labels=labels, training=True
            )

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss

    @tf.function
    def _train_step(self, example):
        """Perform the training step using the distribution strategy."""
        per_replica_loss = self._distribute_strategy.experimental_run_v2(
            self.train_step, args=(example[0], example[1])
        )
        return self._reduce(per_replica_loss, tf.distribute.ReduceOp.SUM)

    def call(
        self,
        training_set: tf.data.Dataset,
        validation_set: tf.data.Dataset,
        log_freq: int = 10,
        measure_performance_freq: int = 10,
    ):
        """
        Start the training.

        Args:
            training_set (:py:obj:`tf.data.Dataset`): Training dataset.
            validation_set (:py:obj:`tf.data.Dataset`): Validation dataset.
            log_freq (int): Specifies how many steps to run before logging the losses,
                e.g. `log_frequency=10` logs every 10 steps of training.
                Pass `log_frequency<=0` in case you don't want to log.
            measure_performance_freq (int): Specifies how many steps to run before
                measuring the performance, e.g. `measure_performance_freq=10`
                measures performance every 10 steps of training.
                Pass `measure_performance_freq<=0` in case you don't want to measure
                performance.

        """
        # set the context properties
        self._context.training_set = training_set
        self._context.validation_set = validation_set

        current_epoch = self._current_epoch()
        self._update_global_batch_size(training_set, self._loss)

        # measure performance on the validation set
        with self._eval_summary_writer.as_default():
            self._context.dataset = validation_set
            self._measure_performance()

        # need to use the global batch size in the training set
        training_set = wrap(
            training_set.unbatch().batch(
                self._global_batch_size, drop_remainder=tf.distribute.has_strategy()
            )
        )

        with self._train_summary_writer.as_default():

            # notify on train start
            self._on_train_start()

            for _ in tf.range(current_epoch, self._epochs):
                distribute_dataset = self._distribute_strategy.experimental_distribute_dataset(
                    training_set
                )

                # notify on epoch start
                self._on_epoch_start()

                for example in distribute_dataset:

                    self._context.current_batch = self.local_example(example, (1, 1))

                    # notify on batch start
                    self._on_batch_start()

                    # perform training step
                    loss = self._train_step(example)

                    # increase global step
                    self._global_step.assign_add(1)

                    # log loss if needed
                    if log_freq > 0 and tf.equal(
                        tf.math.mod(self._global_step, log_freq), 0
                    ):
                        tf.print(f"[{self._global_step.numpy()}] loss: {loss}")

                    # measure performance
                    # this can also be moved to on_batch_end
                    self._measure_performance_if_needed(
                        example, measure_performance_freq
                    )

                    # notify on batch end
                    self._on_batch_end()

                # notify on epoch end
                self._on_epoch_end()

                with self._eval_summary_writer.as_default():
                    self._context.dataset = validation_set
                    self._measure_performance()

            # final callback
            self._on_train_end()
