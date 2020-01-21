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

"""Test LogImageGANCallback."""

from ashpy.callbacks.events import Event
from ashpy.callbacks.gan import LogImageGANCallback

from tests.utils.fake_training_loop import fake_adversarial_training_loop


def test_callbacks(tmpdir):
    """Test the integration between callbacks and trainer."""
    callbacks = [LogImageGANCallback(event=Event.ON_BATCH_END, event_freq=1)]
    fake_adversarial_training_loop(tmpdir, callbacks=callbacks)
