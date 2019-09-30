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
Callbacks in order to gain control over the training loop.

A callback is a set of functions to be called at given stages of the training procedure.
You can use callbacks to implement logging, measure custom metrics or get insight about the
training procedure.
You can pass a list of callbacks (derived from :py:class:`ashpy.callbacks.callback.Callback`)
(as the keyword argument callbacks)
to the `.call()` method of the Trainer.
The relevant methods of the callbacks will then be called at each stage of the training.

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

The basic class is :py:class:`ashpy.callbacks.callback.Callback` .
All possible events as listed as Enum inside :py:class:`ashpy.callbacks.events.Event` .

.. currentmodule:: ashpy.callbacks

.. rubric:: Classes

.. autosummary::
    :nosignatures:
    :toctree: callbacks

    callback.Callback
    counter_callback.CounterCallback
    classifier.LogClassifierCallback
    events.Event
    gan.LogImageGANCallback
    gan.LogImageGANEncoderCallback
    save_callback.SaveCallback
    save_callback.SaveFormat
    save_callback.SaveSubFormat

----

.. rubric:: Modules

.. autosummary::
    :nosignatures:
    :toctree: contexts
    :template: autosummary/submodule.rst

    callback
    classifier
    counter_callback
    events
    gan
    save_callback

"""

from ashpy.callbacks.callback import Callback
from ashpy.callbacks.classifier import LogClassifierCallback
from ashpy.callbacks.counter_callback import CounterCallback
from ashpy.callbacks.events import Event
from ashpy.callbacks.gan import LogImageGANCallback, LogImageGANEncoderCallback
from ashpy.callbacks.save_callback import SaveCallback, SaveFormat, SaveSubFormat

__ALL__ = [
    "Callback",
    "Event",
    "LogClassifierCallback",
    "LogImageGANCallback",
    "LogImageGANEncoderCallback",
    "LogClassifierCallback",
    "SaveCallback",
    "SaveFormat",
    "SaveSubFormat",
]
