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
Collection of Losses.

.. currentmodule:: ashpy.losses

.. rubric:: Executor

.. autosummary::
    :nosignatures:
    :toctree: executor

    executor.Executor
    executor.SumExecutor

----

.. rubric:: Classifier

.. autosummary::
    :nosignatures:
    :toctree: classifier

    classifier.ClassifierLoss

----

.. rubric:: GAN

.. autosummary::
    :nosignatures:
    :toctree: gan

    gan.GANExecutor
    gan.AdversarialLossType
    gan.GeneratorAdversarialLoss
    gan.DiscriminatorAdversarialLoss
    gan.GeneratorBCE
    gan.GeneratorLSGAN
    gan.GeneratorL1
    gan.GeneratorHingeLoss
    gan.FeatureMatchingLoss
    gan.CategoricalCrossEntropy
    gan.Pix2PixLoss
    gan.Pix2PixLossSemantic
    gan.EncoderBCE
    gan.DiscriminatorMinMax
    gan.DiscriminatorLSGAN
    gan.DiscriminatorHingeLoss
    gan.get_adversarial_loss_discriminator
    gan.get_adversarial_loss_generator

----

.. rubric:: Modules

.. autosummary::
    :nosignatures:
    :toctree: losses
    :template: autosummary/submodule.rst

    classifier
    executor
    gan

"""

from ashpy.losses.classifier import ClassifierLoss
from ashpy.losses.executor import Executor, SumExecutor
from ashpy.losses.gan import (
    AdversarialLossType,
    CategoricalCrossEntropy,
    DiscriminatorAdversarialLoss,
    DiscriminatorLSGAN,
    DiscriminatorMinMax,
    EncoderBCE,
    FeatureMatchingLoss,
    GANExecutor,
    GeneratorAdversarialLoss,
    GeneratorBCE,
    GeneratorL1,
    GeneratorLSGAN,
    Pix2PixLoss,
    Pix2PixLossSemantic,
    get_adversarial_loss_discriminator,
    get_adversarial_loss_generator,
)

__ALL__ = [
    "DiscriminatorAdversarialLoss",
    "GeneratorAdversarialLoss",
    "AdversarialLossType",
    "CategoricalCrossEntropy",
    "ClassifierLoss",
    "ClassifierLoss",
    "DiscriminatorLSGAN",
    "DiscriminatorMinMax",
    "DiscriminatorHingeLoss",
    "EncoderBCE",
    "Executor",
    "FeatureMatchingLoss",
    "GANExecutor",
    "GeneratorBCE",
    "GeneratorL1",
    "GeneratorLSGAN",
    "GeneratorHingeLoss",
    "get_adversarial_loss_discriminator",
    "get_adversarial_loss_generator",
    "Pix2PixLoss",
    "Pix2PixLossSemantic",
    "SumExecutor",
]
