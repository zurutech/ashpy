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
    ):
        """
        Initialize the LogClassifierCallback.

        Args:
            event: event to consider
            event_freq: frequency of logging

        """
        super(LogClassifierCallback, self).__init__(
            event=event,
            fn=LogClassifierCallback._log_fn,
            name=name,
            event_freq=event_freq,
        )

    @staticmethod
    def _log_fn(context: ClassifierContext) -> None:
        """
        Log output of the image and label to Tensorboard.

        Args:
            context: current context

        """
        input_tensor = context.current_batch[0]
        out_label = context.current_batch[1]

        log("input_x", input_tensor, context.global_step)
        log("input_y", out_label, context.global_step)
