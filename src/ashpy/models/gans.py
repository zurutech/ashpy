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
GANs Models.

.. currentmodule:: ashpy.models.gans

.. rubric:: Generators

.. autosummary::
    :nosignatures:
    :toctree: models

    ConvGenerator
    DenseGenerator

----

.. rubric:: Discriminators

.. autosummary::
    :nosignatures:
    :toctree: models

    ConvDiscriminator
    DenseDiscriminator

----

.. rubric:: Encoders

.. autosummary::
    :nosignatures:
    :toctree: models

    ConvEncoder
    DenseEncoder

"""
from ashpy.models.convolutional.decoders import Decoder as ConvDecoder
from ashpy.models.convolutional.encoders import Encoder as ConvEncoder
from ashpy.models.fc.decoders import Decoder as DenseDecoder
from ashpy.models.fc.encoders import Encoder as DenseEncoder

ConvGenerator = ConvDecoder
ConvDiscriminator = ConvEncoder
ConvEncoder = ConvEncoder

DenseGenerator = DenseDecoder
DenseDiscriminator = DenseEncoder
DenseEncoder = DenseEncoder
