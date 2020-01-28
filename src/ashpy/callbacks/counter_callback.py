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
Counter Callback implementation.

Callback that count events and calls the passed fn evert event_freq.
"""

from typing import Callable

import tensorflow as tf
from ashpy.callbacks import Callback
from ashpy.callbacks.events import Event
from ashpy.contexts import Context

__ALL__ = ["CounterCallback"]


class CounterCallback(Callback):
    """
    Count events of a specific type. Calls fn passing the context every event_freq.

    Useful for logging or for measuring performance.
    If you want to implement a callback defining a certain behaviour every n_events
    you can just inherit from CounterCallback.
    """

    def __init__(
        self, event: Event, fn: Callable, name: str, event_freq: int = 1
    ) -> None:
        """
        Initialize the CounterCallback.

        Args:
            event (:py:class:`ashpy.events.Event`): event to count.
            fn (:py:class:`Callable`): function to call every `event_freq` events.
            event_freq (int): event frequency.
            name (str): name of the Callback.

        Raises:
            ValueError: if `event_freq` is not valid.

        """
        super().__init__(name=name)
        if not isinstance(event, Event):
            raise TypeError("Use the Event enum!")
        self._event = event

        if event_freq <= 0:
            raise ValueError(
                f"CounterCallback: event_freq cannot be <= 0. Received event_freq = {event_freq}"
            )

        self._event_freq = event_freq
        self._fn = fn
        self._event_counter = tf.Variable(
            0, name=f"{name}event_counter", trainable=False, dtype=tf.int64
        )

    def on_event(self, event: Event, context: Context):
        """
        Count events and calls fn.

        Args:
            event (:py:class:`ashpy.callbacks.events.Event`): current event.
            context (:py:class:`ashpy.contexts.context.Context`): current context.

        """
        # Check the event type
        if event == self._event:

            # Increment event counter
            self._event_counter.assign_add(1)

            # If the module between the event counter and the
            # Frequency is zero, call the fn
            if tf.equal(tf.math.mod(self._event_counter, self._event_freq), 0):
                self._fn(context)
