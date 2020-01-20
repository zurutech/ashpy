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

"""Test a Custom Callback."""

import pytest
from ashpy.callbacks import Callback
from ashpy.callbacks.events import Event

from tests.utils.fake_training_loop import fake_training_loop


class MCallback(Callback):
    """
    Custom callback definition.

    Check the number of times the on_event is triggered.
    """

    def __init__(self, event):
        super(MCallback, self).__init__()
        self._event = event
        self.counter = 0

    def on_event(self, event, context):
        """Simply increment the counter if the event is correct."""
        if event == self._event:
            self.counter += 1


def get_n_events_from_epochs(
    event: Event, epochs: int, dataset_size: int, batch_size: int
):
    """Return the number of events given epochs, dataset_size and batch_size."""
    if event in [Event.ON_TRAIN_START, Event.ON_TRAIN_END]:
        return 1
    if event in [Event.ON_EPOCH_START, Event.ON_EPOCH_END]:
        return epochs
    if event in [Event.ON_BATCH_START, Event.ON_BATCH_END]:
        return (dataset_size / batch_size) * epochs
    if event in [Event.ON_EXCEPTION]:
        return 0
    raise ValueError("Event not compatible")


@pytest.mark.parametrize("event", list(Event))
def test_custom_callbacks(adversarial_logdir: str, event: Event):
    """Test the integration between a custom callback and a trainer."""
    m_callback = MCallback(event)
    callbacks = [m_callback]

    epochs = 2
    dataset_size = 2
    batch_size = 2

    fake_training_loop(
        adversarial_logdir,
        callbacks=callbacks,
        epochs=epochs,
        dataset_size=dataset_size,
        batch_size=batch_size,
    )

    # assert the number of times the on_event has been called
    assert m_callback.counter == get_n_events_from_epochs(
        event, epochs, dataset_size, batch_size
    )
