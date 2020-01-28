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

"""Callback definition."""
import tensorflow as tf
from ashpy.callbacks.events import Event
from ashpy.contexts import Context

__ALL__ = ["Callback"]


class Callback(tf.Module):
    r"""
    Callback definition.

    Every callback must extend from this class.
    This class defines the basic events.
    Every event takes as input the context in order to use the objects defined.
    Inheritance from :py:class:`tf.Module` is required since callbacks have a state

    Order:
    .. code-block::

        --on_train_start

        ----on_epoch_start

        ------on_batch_start

        ------on_batch_end

        ----on_epoch_end

        --on_train_end

        on_exception – if an Exception was raised

        on_event - Called when an event is triggered

    """

    def __init__(self, name: str) -> None:
        """
        Initialize the Callback.

        Args:
            name (str): Callback name.

        Warning:
            When using multiple callbacks with the same trainer make sure they have
            different ids.

        """
        self._name = name

    @property
    def name(self):
        """Return the name of the callback."""
        return self._name

    def on_event(self, event: Event, context: Context) -> None:
        """
        Handle the on_event event.

        Method called when an event is triggered.

        Args:
            event (:py:class:`ashpy.callbacks.events.Event`): triggered event
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """

    def on_train_start(self, context: Context) -> None:
        """
        Handle on_train_start event.

        Method called at the beginning of the training loop.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_TRAIN_START, context)

    def on_train_end(self, context: Context) -> None:
        """
        Handle on_train_end event.

        Method called at the end of the training loop.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_TRAIN_END, context)

    def on_epoch_start(self, context: Context) -> None:
        """
        Handle on_epoch_start event.

        Method called at the beginning of an epoch.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_EPOCH_START, context)

    def on_epoch_end(self, context: Context) -> None:
        """
        Handle on_epoch_end event.

        Method called at the end of an epoch.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_EPOCH_END, context)

    def on_batch_start(self, context: Context) -> None:
        """
        Handle on_batch_start event.

        Method called at the beginning of a batch.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_BATCH_START, context)

    def on_batch_end(self, context: Context) -> None:
        """
        Handle on_batch_end event.

        Method called at the end of a batch.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_BATCH_END, context)

    def on_exception(self, context: Context) -> None:
        """
        Handle on_exception event.

        Method called when an exception is raised.

        Args:
            context (:py:class:`ashpy.contexts.context.Context`): training context

        """
        self.on_event(Event.ON_EXCEPTION, context)
