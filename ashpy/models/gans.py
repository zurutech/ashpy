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

    Generator

----

.. rubric:: Discriminators

.. autosummary::
    :nosignatures:
    :toctree: models

    Discriminator

----

.. rubric:: Encoders

.. autosummary::
    :nosignatures:
    :toctree: models

    Encoder

"""
from ashpy.models.convolutional.decoders import BaseDecoder as BaseConvDecoder
from ashpy.models.convolutional.encoders import BaseEncoder as BaseConvEncoder
from ashpy.models.fc.decoders import BaseDecoder as BaseDenseDecoder
from ashpy.models.fc.encoders import BaseEncoder as BaseDenseEncoder

ConvGenerator = BaseConvDecoder
ConvDiscriminator = BaseConvEncoder
ConvEncoder = BaseConvEncoder

DenseGenerator = BaseDenseDecoder
DenseDiscriminator = BaseDenseEncoder
DenseEncoder = BaseDenseEncoder
