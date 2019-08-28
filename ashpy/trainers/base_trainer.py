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
from abc import ABC, abstractmethod

import tensorflow as tf

from ashpy.losses.executor import Executor
from ashpy.modes import LogEvalMode


class BaseTrainer(ABC):
    r""":py:class:`BaseTrainer` provide an interface for all trainers to inherit from."""

    def __init__(
        self,
        epochs,
        logdir=os.path.join(os.getcwd(), "log"),
        log_eval_mode=LogEvalMode.TEST,
        global_step=tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64),
        post_process_callback=None,
    ):
        r"""
        Primitive trainer interface. Handles model saving and restore.

        Args:
            epochs (int): Number of training epochs.
            logdir (str): Checkpoint and log directory.
            log_eval_mode: models' mode to use when evaluating and logging.
            global_step: tf.Variable that keeps track of the training steps.
            post_process_callback: the function to postprocess the model output, if needed.
        """

        self._distribute_strategy = tf.distribute.get_strategy()
        self._post_process_callback = post_process_callback

        self._epochs = epochs
        self._global_step = global_step
        self._steps_per_epoch = tf.Variable(
            -1, name="steps_per_epoch", trainable=False, dtype=tf.int64
        )
        self._ckpt = tf.train.Checkpoint()
        self._ckpt.objects = []
        self._ckpt.objects.extend([self._global_step, self._steps_per_epoch])
        self._logdir = logdir
        self._manager = tf.train.CheckpointManager(
            self._ckpt, os.path.join(self._logdir, "ckpts"), max_to_keep=3
        )

        self._metrics = []
        self._train_summary_writer = tf.summary.create_file_writer(
            os.path.join(self._logdir, "train")
        )
        self._eval_summary_writer = tf.summary.create_file_writer(
            os.path.join(self._logdir, "eval")
        )
        self._test_summary_writer = tf.summary.create_file_writer(
            os.path.join(self._logdir, "test")
        )

        # Initialize the global batch size to a negative number
        # This is used when a distribution strategy changes the batch size
        self._global_batch_size = -1.0

        self._log_eval_mode = log_eval_mode

    def _log(self, name, out):
        """
        Log the out tensor using name as its name in tensorboard.

        Args:
            name: summary name.
            out: the tensor to log.

        """
        rank = tf.rank(out)
        step = self._global_step

        # handle post post_process_callback
        if self._post_process_callback:
            out = self._post_process_callback(out)

        # log
        if tf.equal(rank, 4):
            # tensorboard 2.0 does not support float images in [-1, 1]
            # only in [0,1]
            if self._post_process_callback is None and out.dtype == tf.float32:
                # TODO: the hypothesis is that image are in [-1,1] how to check?
                out = (out + 1.0) / 2

            tf.summary.image(
                name, out, max_outputs=tf.math.minimum(tf.shape(out)[0], 16), step=step
            )
        if tf.equal(rank, 2):
            tf.summary.histogram(name, out, step=step)

    def _update_global_batch_size(self, dataset, executors=None):
        """Given a dataset and the current distribution strategy sets the
        self._global_batch_size variable where needed.
        Args:
            dataset: a dataset from wich the batch size will be extracted.
            executors: a list of executor with the property "global_batch_size" settable.
        """

        sample = next(iter(dataset.take(1)))
        if isinstance(sample, tuple):
            if isinstance(sample[0], tuple):
                sample = sample[0]
            batch_size_per_replica = sample[0].shape[0]
        elif isinstance(sample, tf.Tensor):
            batch_size_per_replica = sample.shape[0]
        else:
            raise ValueError("Unable to extract the batch size from the dataset")
        self._global_batch_size = (
            self._distribute_strategy.num_replicas_in_sync * batch_size_per_replica
        )

        if executors:
            if isinstance(executors, list):
                for executor in executors:
                    executor.global_batch_size = self._global_batch_size
            if isinstance(executors, Executor):
                executors.global_batch_size = self._global_batch_size

    def _reduce(self, per_replica_tensor, reduce_op):
        """Given the input tensor, reduces it in a distributed fashion, using the specified op."""
        context = tf.distribute.get_replica_context()
        if context:
            return context.all_reduce(reduce_op, per_replica_tensor)
        return self._distribute_strategy.reduce(
            reduce_op, per_replica_tensor, axis=None
        )

    def _restore_or_init(self):
        """Restores or initializes the persistence layer (checkpoint)."""
        if self._manager.latest_checkpoint:
            self._ckpt.restore(self._manager.latest_checkpoint)
            print(f"Restored checkpoint {self._manager.latest_checkpoint}.")
        else:
            print("Initializing checkpoint.")

    def _save(self):
        """Save the current checkpointable object status."""
        ckpt = self._manager.save()
        # print is captured from pydoc - deterministic output can be used
        # to run tests.
        print(f"[{self._global_step.numpy()}] Saved checkpoint: {ckpt}")

    def _current_epoch(self):
        """
        Get the current epoch using the (restored) variables.

        Returns:
            current_epoch (int)

        """
        current_epoch = tf.constant(0, dtype=tf.int64)
        if tf.math.greater(self._steps_per_epoch, tf.constant(0, dtype=tf.int64)):
            current_epoch = tf.cast(
                tf.math.floor(self._global_step / self._steps_per_epoch), tf.int64
            )
        return current_epoch

    def _epoch_completed(self, epoch):
        """
        Handle the end of the training epoch.

        Args:
            epoch (int): the just completed training epoch.

        """
        if tf.math.less(self._steps_per_epoch, 0):
            # only the first time, save the number of steps per epoch
            self._steps_per_epoch.assign(self._global_step)
        self._save()
        print(f"Epoch {epoch} completed.")

    def _log_metrics_and_reset(self):
        step = self._global_step.numpy()

        for metric_obj in self._metrics:
            metric_obj.log(step=step)
            metric_obj.reset_states()

    def _dataset_from_example(self, example, dims):
        columns = []
        for idx, dim in enumerate(dims):
            if dim > 1:
                columns.append(
                    tuple(
                        tf.concat(
                            self._distribute_strategy.experimental_local_results(
                                example[idx][inner]
                            ),
                            axis=0,
                        )
                        for inner in range(dim)
                    )
                )
            else:
                columns.append(
                    tf.concat(
                        self._distribute_strategy.experimental_local_results(
                            example[idx]
                        ),
                        axis=0,
                    )
                )
        return tf.data.Dataset.from_tensor_slices(tuple(columns))

    @abstractmethod
    def call(self, dataset: tf.data.Dataset):
        """
        Execute the training process.

        Iterate over the elements of a :py:class:`tf.data.Dataset`.
        The dataset must contain everything needed to train the model.

        Args:
            dataset: A :py:class:`tf.data.Dataset` to loop on to train the model.
        """

    def __call__(self, *args, **kwargs):
        """Invoke the trainer."""
        self.call(*args, **kwargs)
