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

import os

import tensorflow as tf

from ashpy.contexts.classifier import ClassifierContext
from ashpy.datasets import wrap
from ashpy.metrics import ClassifierLoss
from ashpy.trainers.base_trainer import BaseTrainer


class ClassifierTrainer(BaseTrainer):
    r""":py:class:`ClassifierTrainer` provide the standard training loop for a classifier."""

    def __init__(
        self,
        model,
        optimizer,
        loss,
        epochs,
        metrics=None,
        logdir=os.path.join(os.getcwd(), "log"),
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        post_process_callback=None,
    ):
        r"""
        Instantiate the :py:class:`ClassifierTrainer` trainer.

        Args:
            model (:py:class:`tf.keras.Model`): A :py:class:`tf.keras.Model` model.
            optimizer (:py:class:`tf.optimizers.Optimizer`): A
                :py:class:`tf.optimizers.Optimizer`.
            loss (:obj:`callable`): A loss function built following :py:mod:`tf.losses`.
            epochs (int): Number of training epochs.
            metrics: (List): List of python objects (dictionaries or tf.metrics objects) to
                measure on training and validation data.
            logdir (str): Checkpoint and log directory.
            global_step: tf.Variable that keeps track of the training steps.
            post_process_callback(:obj:`callable`): the function to postprocess the model output,
                if needed.

        Examples:
            .. testcode::
                import shutil
                import operator
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

                metrics = [
                    ClassifierMetric(tf.metrics.Accuracy()),
                    ClassifierMetric(tf.metrics.BinaryAccuracy()),
                ]

                trainer = ClassifierTrainer(model, optimizer, loss, epochs, metrics, logdir=logdir)
                train, validation = toy_dataset(), toy_dataset()
                trainer(train, validation)
                shutil.rmtree(logdir)

            .. testoutput::

                Initializing checkpoint.
                [500] Saved checkpoint: testlog/ckpts/ckpt-1
                Epoch 1 completed.
                [1000] Saved checkpoint: testlog/ckpts/ckpt-2
                Epoch 2 completed.

        """
        super().__init__(
            epochs=epochs,
            logdir=logdir,
            global_step=global_step,
            post_process_callback=post_process_callback,
        )

        self._model = model
        self._optimizer = optimizer
        self._loss = loss

        self._avg_loss = ClassifierLoss()
        if metrics:
            metrics.append(self._avg_loss)
        else:
            metrics = [self._avg_loss]

        for metric in metrics:
            metric.logdir = self._logdir
        self._metrics = metrics

        self._ckpt.objects.extend([self._optimizer, self._model])
        self._restore_or_init()
        self._context = ClassifierContext(
            classifier_model=self._model,
            log_eval_mode=self._log_eval_mode,
            global_step=self._global_step,
            ckpt=self._ckpt,
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
        """The training step that uses the distribution strategy."""
        per_replica_loss = self._distribute_strategy.experimental_run_v2(
            self.train_step, args=(example[0], example[1])
        )
        return self._reduce(per_replica_loss, tf.distribute.ReduceOp.SUM)

    def _measure_performance(self, dataset):
        """Measure and log metrics on the dataset."""
        context = ClassifierContext(
            self._model,
            self._loss,
            dataset,
            self._metrics,
            log_eval_mode=self._log_eval_mode,
            global_step=self._global_step,
            ckpt=self._ckpt,
        )
        context.measure_metrics()
        context.model_selection()
        self._log_metrics_and_reset()

    def call(self, train_set, validation_set):
        """
        Start the training.

        Args:
            train_set (:py:obj:`tf.data.Dataset`): Training dataset.
            validation_set (:py:obj:`tf.data.Dataset`): Validation dataset.
        """
        current_epoch = self._current_epoch()
        self._update_global_batch_size(train_set, self._loss)
        with self._eval_summary_writer.as_default():
            self._measure_performance(validation_set)

        # need to use the global batch size in the training set
        train_set = wrap(
            train_set.unbatch().batch(
                self._global_batch_size, drop_remainder=tf.distribute.has_strategy()
            )
        )

        with self._train_summary_writer.as_default():
            for epoch in tf.range(current_epoch, self._epochs):
                distribute_dataset = self._distribute_strategy.experimental_distribute_dataset(
                    train_set
                )

                for example in distribute_dataset:
                    loss = self._train_step(example)
                    self._global_step.assign_add(1)
                    if tf.equal(tf.math.mod(self._global_step, 10), 0):
                        tf.print(f"[{self._global_step.numpy()}] loss: {loss}")
                        self._measure_performance(
                            self._dataset_from_example(example, (1, 1)).batch(
                                self._global_batch_size
                            )
                        )
                        self._log("input_x", example[0])
                        self._log("input_y", example[1])

                self._epoch_completed(epoch + 1)
                with self._eval_summary_writer.as_default():
                    self._measure_performance(validation_set)
