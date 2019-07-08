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
The executor.
An object that carries a function and the way of executing it.
Given a context.
"""

import abc

import tensorflow as tf


class Executor:
    """Carry a function and the wa of executing it. Given a context."""

    def __init__(self, fn=None):
        """
        Initialize the Executor.

        Args:
            fn (:py:class:`tf.keras.losses.Loss`): A Keras Loss to execute.

        Returns:
            :py:obj:`None`

        """
        if fn is not None:
            assert isinstance(fn, tf.keras.losses.Loss)
            self._fn = fn
            # we always work as in a strategy context
            self._fn.reduction = tf.keras.losses.Reduction.NONE
        self._distribute_strategy = tf.distribute.get_strategy()
        self._global_batch_size = -1
        self._weight = lambda _: 1.0

    @property
    def weight(self):
        return self._weight

    @property
    def fn(self):
        """Retrieve the function to execute."""
        return self._fn

    @staticmethod
    def reduce_loss(call_fn):
        """
        Create a Decorator to reduce Losses. Use to simplify things.

        Apply a ``reduce sum`` operation to the loss and divide the result
        by the batch size.

        Args:
            call_fn: the executor call method

        Return:
            :py:func: The function decorated

        """
        # decorator definition
        def _reduce(self, *args, **kwargs):
            return tf.nn.compute_average_loss(
                call_fn(self, *args, **kwargs),
                global_batch_size=self._global_batch_size,
            )

        return _reduce

    @property
    def global_batch_size(self):
        """Return the global batch size."""
        return self._global_batch_size

    @global_batch_size.setter
    def global_batch_size(self, global_batch_size):
        r"""
        Set the `_global_batch_size` property.

        Args:
            global_batch_size (int): Global batch size. In the case of a distributed
                setup this is `batch_size on GPU * n. of GPUs`.

        Return:
            :py:obj:`None`

        """
        assert global_batch_size > 0
        self._global_batch_size = global_batch_size

    @abc.abstractmethod
    def call(self, context, **kwargs):
        r"""
        Execute the function, using the information provided by the context.

        Args:
            context (:py:class:`ashpy.contexts.BaseContext`): The function execution Context.
        """

    def __call__(self, context, **kwargs):
        r"""
        Invoke the function using the Context.

        Args:
            context (:py:class:`ashpy.contexts.BaseContext`): The function execution Context.

        """
        return self._weight(context.global_step) * self.call(context, **kwargs)

    def __add__(self, other):
        if isinstance(other, SumExecutor):
            other_executors = other.executors
        else:
            other_executors = [other]

        all_executors = [self] + other_executors
        return SumExecutor(all_executors)

    def __mul__(self, other):
        assert isinstance(other, (float, int, tf.Tensor)) or callable(other)
        weight = self._weight
        if callable(other):
            self._weight = lambda step: weight(step) * other(step)
        else:
            self._weight = lambda step: weight(step) * other
        return self

    def __rmul__(self, other):
        return self * other


class SumExecutor(Executor):
    """
    The sum executor. Executes the call of each fn and weights the losses.

    Each Executor gets called (thus reducing its carried function), the results are
    then summed together.
    """

    # TODO: Add reference to Pix2Pix Loss in the losses.gans

    def __init__(self, executors):
        """
        Initialize the SumExecutor.

        Args:
            executors (:py:obj:`list` of :py:class:`ashpy.executors.Executor`): Array of
                :py:obj:`ashpy.executors.Executor` to sum evaluate and sum together.

        Returns:
            :py:obj:`None`

        """
        super().__init__()
        self._executors = executors
        self._global_batch_size = 1

    @property
    def executors(self):
        """Return the array of Executors."""
        return self._executors

    @Executor.global_batch_size.setter
    def global_batch_size(self, global_batch_size):
        assert global_batch_size > 0
        self._global_batch_size = global_batch_size
        for executor in self._executors:
            executor.global_batch_size = global_batch_size

    def call(self, *args, **kwargs):
        """Evaluate and sum together the Executors."""
        result = tf.add_n([executor(*args, **kwargs) for executor in self._executors])
        return result

    def __add__(self, other):
        if isinstance(other, SumExecutor):
            executors = other.executors
        else:
            executors = [other.executors]

        all_executors = self.executors + executors
        return SumExecutor(all_executors)
