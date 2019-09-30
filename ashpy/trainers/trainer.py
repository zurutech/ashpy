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
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from ashpy.callbacks import Callback
from ashpy.contexts import Context
from ashpy.losses.executor import Executor
from ashpy.metrics import Metric
from ashpy.modes import LogEvalMode
from ashpy.utils.utils import validate_objects


class Trainer(ABC):
    r""":py:class:`Trainer` provide an interface for all trainers to inherit from."""

    def __init__(
        self,
        epochs: int,
        example_dim: Tuple[int, int],
        logdir: str = os.path.join(os.getcwd(), "log"),
        log_eval_mode: LogEvalMode = LogEvalMode.TEST,
        global_step: Optional[tf.Variable] = None,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        r"""
        Primitive trainer interface. Handles model saving and restore.

        Args:
            epochs (int): Number of training epochs.
            example_dim (Tuple[int, int]): Dimension of an example. In the case of GANs
                the example has dimension (2,1) since it's composed by a tuple in which the first
                element is a tuple with 2 components and the second component is a single element.
                In the case of classifier the example has dimension (1, 1) since it's composed
                by the example and the label.
            logdir (str): Checkpoint and log directory.
            log_eval_mode (py:class:`ashpy.modes.LogEvalMode`) models' mode
                to use when evaluating and logging.
            global_step (Optional[py:class:`ashpy.modes.LogEvalMode`]): tf.Variable that
                keeps track of the training steps.
            metrics (Optional[List[:py:class:`ashpy.metrics.Metric`]]): list of metrics.
            callbacks (Optional[List[:py:class:`ashpy.callbacks.Callback`]]): list of callbacks
                to handle events.

        """
        self._distribute_strategy = tf.distribute.get_strategy()
        self._context = Context()

        # set and validate metrics
        if metrics is None:
            metrics = []
        self._metrics = metrics
        self._validate_metrics()

        # set and validate callbacks
        if callbacks is None:
            callbacks = []
        self._callbacks = callbacks
        self._validate_callbacks()

        self._epochs = epochs

        # global step must be created here
        # do not use tf.Variable as default argument
        # since tf.Variable are mutable
        # see https://docs.python-guide.org/writing/gotchas/
        # for more information
        # in short: default arguments are initialized at definition
        # time and not at call time
        if global_step is None:
            global_step = tf.Variable(
                0, name="global_step", trainable=False, dtype=tf.int64
            )
        self._global_step = global_step

        self._steps_per_epoch = tf.Variable(
            -1, name="steps_per_epoch", trainable=False, dtype=tf.int64
        )
        self._checkpoint = tf.train.Checkpoint()
        self._checkpoint.objects = []
        self._checkpoint.objects.extend(
            [self._global_step, self._steps_per_epoch] + self._callbacks
        )
        self._logdir = logdir
        self._manager = tf.train.CheckpointManager(
            self._checkpoint, os.path.join(self._logdir, "ckpts"), max_to_keep=3
        )

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

        self._example_dim = example_dim

    @property
    def context(self) -> Context:
        """Return the training context."""
        return self._context

    @context.setter
    def context(self, _context: Context):
        """
        Set the context.

        Args:
            _context (:py:class:`ashpy.contexts.context.Context`): Context to set.

        """
        self._context = _context

    def _validate_metrics(self):
        """Check if every metric is an :py:class:`ashpy.metrics.Metric`."""
        validate_objects(self._metrics, Metric)

    def _validate_callbacks(self):
        """Check if every callback is an :py:class:`ashpy.callbacks.Callback`."""
        validate_objects(self._callbacks, Callback)

    def _update_global_batch_size(
        self,
        dataset: tf.data.Dataset,
        executors: Optional[Union[List[Executor], Executor]] = None,
    ):
        """
        Set the `self._global_batch_size` variable where needed.

        Args:
            dataset (:py:class:`tf.data.Dataset`): a dataset from which the batch size will be extracted.
            executors (Union[List[:py:class:`ashpy.losses.executor.Executor`],
                :py:class:`ashpy.losses.executor.Executor`]: a list of executor
                with the property "global_batch_size".

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
        """Reduce the input tensor in a distributed fashion, using the specified op."""
        context = tf.distribute.get_replica_context()
        if context:
            return context.all_reduce(reduce_op, per_replica_tensor)
        return self._distribute_strategy.reduce(
            reduce_op, per_replica_tensor, axis=None
        )

    def _restore_or_init(self):
        """Restore or initialize the persistence layer (checkpoint)."""
        if self._manager.latest_checkpoint:
            self._checkpoint.restore(self._manager.latest_checkpoint)
            print(f"Restored checkpoint {self._manager.latest_checkpoint}.")
        else:
            print("Initializing checkpoint.")

    def _save(self):
        """Save the current checkpointable object status."""
        checkpoint = self._manager.save()
        # print is captured from pydoc - deterministic output can be used
        # to run tests.
        print(f"[{self._global_step.numpy()}] Saved checkpoint: {checkpoint}")

    def _current_epoch(self) -> tf.Tensor:
        """
        Get the current epoch using the (restored) variables.

        Returns:
            current_epoch (:py:class:`tf.Tensor`): the current epoch of training.

        """
        current_epoch = tf.constant(0, dtype=tf.int64)
        if tf.math.greater(self._steps_per_epoch, tf.constant(0, dtype=tf.int64)):
            current_epoch = tf.cast(
                tf.math.floor(self._global_step / self._steps_per_epoch), tf.int64
            )
        return current_epoch

    def _log_metrics_and_reset(self):
        """Call for each metric the log and reset_states."""
        step = self._global_step.numpy()

        for metric_obj in self._metrics:
            metric_obj.log(step=step)
            metric_obj.reset_states()

    def measure_metrics(self) -> None:
        """Measure the metrics."""
        for metric in self._metrics:
            metric.update_state(self._context)

    def model_selection(self) -> None:
        """Use the metrics to perform model selection."""
        for metric in self._metrics:
            metric.model_selection(self._checkpoint, self._global_step)

    def _measure_performance_if_needed(
        self, example: tf.Tensor, measure_performance_freq: int
    ):
        """
        Measure performance if needed.

        Measure performance if self._global_step % measure_performance_freq is 0.
        """
        # measure performance if needed
        if measure_performance_freq > 0 and tf.equal(
            tf.math.mod(self._global_step, measure_performance_freq), 0
        ):
            # setup context
            self._context.current_batch = self.local_example(
                example, dims=self._example_dim
            )
            self._context.dataset = self._dataset_from_example(
                example, dims=self._example_dim
            ).batch(self._global_batch_size)

            # measure performance
            self._measure_performance()

    def _measure_performance(self):
        """Measure performance on dataset."""
        self.measure_metrics()
        self.model_selection()
        self._log_metrics_and_reset()

    def _dataset_from_example(self, example, dims) -> tf.data.Dataset:
        """
        Get a dataset from a given example.

        Returns:
            The dataset containing only the example.

        """
        example = self.local_example(example, dims)
        return tf.data.Dataset.from_tensor_slices(example)

    def local_example(self, example, dims):
        """
        Return a local example from a distributed example.

        Returns:
            A local example from a distributed example.

        """
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
        return tuple(columns)

    @abstractmethod
    def call(self, *args, **kwargs):
        """Execute the training process."""

    def __call__(self, *args, **kwargs):
        """Invoke the trainer."""
        try:
            self.call(*args, **kwargs)

        except (Exception, KeyboardInterrupt) as ex:
            self._context.exception = ex
            self._on_exception()
            raise ex

    def _on_train_start(self) -> None:
        """Handle the start of training."""
        for callback in self._callbacks:
            callback.on_train_start(self._context)

    def _on_train_end(self) -> None:
        """Handle the end of training."""
        print(f"Training finished after {self._current_epoch().numpy()} epochs.")
        for callback in self._callbacks:
            callback.on_train_end(self._context)

    def _on_epoch_start(self) -> None:
        """Handle the start of the training epoch."""
        print(f"Starting epoch {self._current_epoch().numpy() + 1}.")
        for callback in self._callbacks:
            callback.on_epoch_start(self._context)

    def _on_epoch_end(self) -> None:
        """Handle the end of the training epoch."""
        if tf.math.less(self._steps_per_epoch, 0):
            # only the first time, save the number of steps per epoch
            self._steps_per_epoch.assign(self._global_step)
        self._save()
        print(f"Epoch {self._current_epoch().numpy()} completed.")
        for callback in self._callbacks:
            callback.on_epoch_end(self._context)

    def _on_batch_start(self) -> None:
        """Handle the start of a training batch."""
        for callback in self._callbacks:
            callback.on_batch_start(self._context)

    def _on_batch_end(self) -> None:
        """Handle the end of a training batch."""
        for callback in self._callbacks:
            callback.on_batch_end(self._context)

    def _on_exception(self) -> None:
        """Handle the exception."""
        for callback in self._callbacks:
            callback.on_exception(self._context)
