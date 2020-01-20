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
The Executor.

An object that, given an :py:class:`ashpy.contexts.Context`, carries a
function and the way of executing it.
"""
from __future__ import annotations

import abc
from typing import Callable, List, Union

import tensorflow as tf


class Executor:
    """Carry a function and the way of executing it. Given a context."""

    def __init__(self, fn: tf.keras.losses.Loss = None) -> None:
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
            # We always work as in a strategy context
            self._fn.reduction = tf.keras.losses.Reduction.NONE
        self._distribute_strategy = tf.distribute.get_strategy()
        self._global_batch_size = -1
        self._weight = lambda _: 1.0

    @property
    def weight(self) -> Callable[..., float]:
        """
        Return the loss weight.

        This weight is multiplied by the loss value.
        This is useful when working with multiples losses.

        Returns:
            :py:obj:`typing.Callable`: Callable returning the weight (:py:obj:`float`).

        """
        return self._weight

    @property
    def fn(self) -> tf.keras.losses.Loss:  # pylint: disable=invalid-name
        """
        Return the Keras loss function to execute.

        Returns:
            :py:obj:`tf.keras.losses.Loss`: Keras Loss.

        """
        return self._fn

    @staticmethod
    def reduce_loss(call_fn: Callable) -> Callable:
        """
        Create a Decorator to reduce Losses. Used to simplify things.

        Apply a ``reduce sum`` operation to the loss and divide the result
        by the batch size.

        Args:
            call_fn (:py:obj:`typing.Callable`): The executor call method.

        Return:
            :py:obj:`typing.Callable`: The decorated function.

        """
        # decorator definition
        def _reduce(self, *args, **kwargs):
            return tf.nn.compute_average_loss(
                call_fn(self, *args, **kwargs),
                global_batch_size=self._global_batch_size,  # pylint: disable=protected-access
            )

        return _reduce

    @property
    def global_batch_size(self) -> int:
        """
        Global batch size comprises the batch size for each cpu.

        Calculated as batch_size_for_replica*replica_numbers.

        Returns:
            :obj:`int`: Global Batch size value.

        """
        return self._global_batch_size

    @global_batch_size.setter
    def global_batch_size(self, global_batch_size) -> None:
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
    def call(self, context, **kwargs) -> tf.Tensor:
        r"""
        Execute the function, using the information provided by the context.

        Args:
            context (:py:class:`ashpy.contexts.Context`): The function
                execution Context.

        Returns:
            :py:obj:`tf.Tensor`: Output Tensor.

        """

    def __call__(self, context, **kwargs) -> tf.Tensor:
        r"""
        Invoke the function using the Context.

        Args:
            context (:py:class:`ashpy.contexts.Context`): The function
                execution Context.

        Returns:
            :py:obj:`tf.Tensor`: Output Tensor.

        """
        return self._weight(context.global_step) * self.call(context, **kwargs)

    def __add__(self, other) -> SumExecutor:
        """Concatenate Executors together into a SumExecutor."""
        if isinstance(other, SumExecutor):
            other_executors = other.executors
        else:
            other_executors = [other]

        all_executors = [self] + other_executors
        return SumExecutor(all_executors)

    def __mul__(self, other: Union[Callable[..., float], float, int, tf.Tensor]):
        """
        Given current weight stored inside the Executor multiplies it by ``other``.

        Args:
            other (Either a :py:obj:`typing.Callable` or :obj:`float`,
                :obj:`int`, :py:class:`tf.Tensor`):
                The value (or function returning it) to use in the multiplication.

        """
        assert isinstance(other, (float, int, tf.Tensor)) or callable(other)
        weight = self._weight
        if isinstance(other, (int, float, tf.Tensor)):
            _other: Union[int, float, tf.Tensor] = other
            self._weight = lambda step: weight(step) * _other
        else:
            __other: Callable[..., float] = other
            self._weight = lambda step: weight(step) * __other(step)
        return self

    def __rmul__(self, other):
        """See `__mul__` method."""
        return self * other


class SumExecutor(Executor):
    """
    The sum executor. Executes the call of each fn and weights the losses.

    Each Executor gets called (thus reducing its carried function), the results are
    then summed together.
    """

    def __init__(self, executors) -> None:
        """
        Initialize the SumExecutor.

        Args:
            executors (:py:obj:`list` of [:py:class:`ashpy.executors.Executor`]): Array of
                :py:obj:`ashpy.executors.Executor` to sum evaluate and sum together.

        Returns:
            :py:obj:`None`

        """
        super().__init__()
        self._executors = executors
        self._global_batch_size = 1

    @property
    def executors(self) -> List[Executor]:
        """Return the List of Executors."""
        return self._executors

    @Executor.global_batch_size.setter  # pylint: disable=no-member
    def global_batch_size(self, global_batch_size: int) -> None:
        """Set global batch size property."""
        assert global_batch_size > 0
        self._global_batch_size = global_batch_size
        for executor in self._executors:
            executor.global_batch_size = global_batch_size

    def call(self, *args, **kwargs) -> tf.Tensor:
        """
        Evaluate and sum together the Executors.

        Returns:
            :py:classes:`tf.Tensor`: Output Tensor.

        """
        result = tf.add_n([executor(*args, **kwargs) for executor in self._executors])
        return result

    def __add__(self, other: Union[SumExecutor, Executor]):
        """Concatenate Executors together into a SumExecutor."""
        if isinstance(other, SumExecutor):
            executors = other.executors
        else:
            executors = [other]

        all_executors = self.executors + executors
        return SumExecutor(all_executors)
