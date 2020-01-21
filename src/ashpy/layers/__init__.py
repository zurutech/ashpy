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
Collection of layers.

.. currentmodule:: ashpy.layers

.. rubric:: Layers

.. autosummary::
    :nosignatures:
    :toctree: layers

    instance_normalization.InstanceNormalization
    attention.Attention

"""

from ashpy.layers.attention import Attention
from ashpy.layers.instance_normalization import InstanceNormalization

__ALL__ = ["Attention", "InstanceNormalization"]
