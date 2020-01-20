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
Collection of Convolutional Models constructors.

.. currentmodule:: ashpy.models.convolutional

.. rubric:: Interfaces

.. autosummary::
    :nosignatures:
    :toctree: convolutional

    interfaces.Conv2DInterface

----

.. rubric:: Decoders

.. autosummary::
    :nosignatures:
    :toctree: convolutional

    decoders.Decoder
    decoders.FCNNDecoder

----

.. rubric:: Encoders

.. autosummary::
    :nosignatures:
    :toctree: convolutional


    encoders.Encoder
    encoders.FCNNEncoder

----

.. rubric:: Autoencoders

.. autosummary::
    :nosignatures:
    :toctree: fc


    autoencoders.Autoencoder
    autoencoders.FCNNAutoencoder

----

.. rubric:: UNet

.. autosummary::
    :nosignatures:
    :toctree: convolutional

    unet.UNet
    unet.SUNet
    unet.FUNet

----

.. rubric:: Discriminators

.. autosummary::
    :nosignatures:
    :toctree: convolutional

    discriminators.MultiScaleDiscriminator
    discriminators.PatchDiscriminator

----

.. rubric:: Pix2PixHD

.. autosummary::
    :nosignatures:
    :toctree: convolutional

    pix2pixhd.LocalEnhancer
    pix2pixhd.GlobalGenerator

----

.. rubric:: Modules

.. autosummary::
    :nosignatures:
    :toctree: convolutional
    :template: autosummary/submodule.rst

    autoencoders
    discriminators
    decoders
    encoders
    interfaces
    unet
    pix2pixhd

"""
