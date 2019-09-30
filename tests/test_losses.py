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
Test losses inside the AdversarialLossType Enum
"""
import pytest
import tensorflow as tf

from ashpy.losses.gan import (
    AdversarialLossType,
    get_adversarial_loss_discriminator,
    get_adversarial_loss_generator,
)
from ashpy.models.gans import ConvDiscriminator, ConvGenerator
from ashpy.trainers import AdversarialTrainer
from tests.utils.fake_training_loop import fake_training_loop


@pytest.mark.parametrize("loss_type", list(AdversarialLossType))
def test_losses(loss_type: AdversarialLossType, adversarial_logdir: str):
    """
    Test the integration between losses and trainer
    """

    # Losses
    generator_loss = get_adversarial_loss_generator(loss_type)()
    discriminator_loss = get_adversarial_loss_discriminator(loss_type)()

    fake_training_loop(
        adversarial_logdir,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
    )
