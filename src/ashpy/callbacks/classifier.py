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

"""Classifier callbacks."""
import tensorflow as tf
from ashpy.callbacks.counter_callback import CounterCallback
from ashpy.callbacks.events import Event
from ashpy.contexts import ClassifierContext
from ashpy.utils.utils import log


class LogClassifierCallback(CounterCallback):
    """
    Callback used for logging Classifier images to Tensorboard.

    Logs the Input Image and true label.
    """

    def __init__(
        self,
        event: Event = Event.ON_EPOCH_END,
        name="log_classifier_callback",
        event_freq: int = 1,
        input_is_zero_centered: bool = True,
    ):
        """
        Initialize the LogClassifierCallback.

        Args:
            name: name of the callback
            event: event to consider
            event_freq: frequency of logging
            input_is_zero_centered: if True, the callback assumes the input is in [-1, 1] if it
                is an image with type tf.float. If False, the callback assumes the input is [0,
                1] if type float, and [0, 255] if type is uint. If the input type is float and
                the image is in [0, 1] use False. If the input type is uint this parameter is
                ignored.
        """
        super(LogClassifierCallback, self).__init__(
            event=event, fn=self._log_fn, name=name, event_freq=event_freq,
        )
        self._input_is_zero_centered = input_is_zero_centered

    def _log_fn(self, context: ClassifierContext) -> None:
        """
        Log output of the image and label to Tensorboard.

        Args:
            context: current context

        """
        input_tensor = context.current_batch[0]
        out_label = context.current_batch[1]

        rank = tf.rank(input_tensor)

        # if it is an image scale it if needed
        if (
            tf.equal(rank, 4)
            and (input_tensor.dtype == tf.float32 or input_tensor.dtype == tf.float64)
            and self._input_is_zero_centered
        ):
            input_tensor = (input_tensor + 1) / 2

        log("input_x", input_tensor, context.global_step)
        log("input_y", out_label, context.global_step)
