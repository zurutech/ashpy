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

"""Test CounterCallback."""

import pytest
import tensorflow as tf
from ashpy.callbacks import CounterCallback, Event
from ashpy.models.gans import ConvDiscriminator, ConvGenerator

from tests.utils.fake_training_loop import fake_training_loop


class FakeCounterCallback(CounterCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_counter = 0

    def on_event(self, event, context):
        if event == self._event:
            self.fake_counter += 1
        super().on_event(event, context)


@pytest.fixture()
def _models():
    image_resolution = (28, 28)
    layer_spec_input_res = (7, 7)
    layer_spec_target_res = (7, 7)
    kernel_size = 5
    channels = 1

    # Model definition
    generator = ConvGenerator(
        layer_spec_input_res=layer_spec_input_res,
        layer_spec_target_res=image_resolution,
        kernel_size=kernel_size,
        initial_filters=32,
        filters_cap=16,
        channels=channels,
    )

    discriminator = ConvDiscriminator(
        layer_spec_input_res=image_resolution,
        layer_spec_target_res=layer_spec_target_res,
        kernel_size=kernel_size,
        initial_filters=16,
        filters_cap=32,
        output_shape=1,
    )

    return generator, discriminator


def test_counter_callback_multiple_events():
    """Counter Callback should not receive multiple events."""
    with pytest.raises(TypeError):
        clbk = FakeCounterCallback(
            event=[Event.ON_EPOCH_END],
            name="TestCounterCallbackMultipleEvents",
            fn=lambda context: print("Bloop"),
        )


# TODO: parametrize tests following test_save_callback.py
def test_counter_callback(_models, adversarial_logdir):
    clbk = FakeCounterCallback(
        event=Event.ON_EPOCH_END,
        name="TestCounterCallback",
        fn=lambda context: print("Bloop"),
    )
    callbacks = [clbk]
    generator, discriminator = _models
    fake_training_loop(
        adversarial_logdir,
        callbacks=callbacks,
        generator=generator,
        discriminator=discriminator,
        epochs=1,
    )
    assert clbk.fake_counter == 1
