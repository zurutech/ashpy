# Copyright 2020 Zuru Tech HK Limited. All Rights Reserved.
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
Restorers allow for easy restoration of tracked objects from :class:`tf.train.Checkpoint`.

.. currentmodule:: ashpy.restorers

.. rubric:: Classes

.. autosummary::
    :nosignatures:
    :toctree: restorers

    Restorer
    AdversarialRestorer
    AdversarialEncoderRestorer
    ClassifierRestorer

----

.. rubric:: Modules

.. autosummary::
    :nosignatures:
    :toctree: restorers
    :template: autosummary/submodule.rst

    restorer
    classifier
    gan

"""

from ashpy.restorers.classifier import ClassifierRestorer
from ashpy.restorers.gan import AdversarialEncoderRestorer, AdversarialRestorer
from ashpy.restorers.restorer import Restorer

__ALL__ = [
    "Restorer",
    "AdversarialRestorer",
    "AdversarialEncoderRestorer",
    "ClassifierRestorer",
]
