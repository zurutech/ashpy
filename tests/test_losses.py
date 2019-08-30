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
import shutil

import pytest
import tensorflow as tf

from ashpy.models.gans import ConvGenerator, ConvDiscriminator
from ashpy.losses.gan import (
    AdversarialLossType,
    get_adversarial_loss_generator,
    get_adversarial_loss_discriminator,
)
from ashpy.trainers import AdversarialTrainer


@pytest.mark.parametrize("loss_type", list(AdversarialLossType))
def test_losses(loss_type: AdversarialLossType, adversarial_logdir: str):
    """
    Test the integration between losses and trainer
    """
    # test parameters
    image_resolution = (28, 28)
    kernel_size = (5, 5)
    batch_size = 2
    dataset_size = 100
    latent_dim = 100

    # model definition
    generator = ConvGenerator(
        layer_spec_input_res=(7, 7),
        layer_spec_target_res=image_resolution,
        kernel_size=kernel_size,
        initial_filters=32,
        filters_cap=16,
        channels=1,
    )

    discriminator = ConvDiscriminator(
        layer_spec_input_res=image_resolution,
        layer_spec_target_res=(7, 7),
        kernel_size=kernel_size,
        initial_filters=16,
        filters_cap=32,
        output_shape=1,
    )

    # Losses
    generator_loss = get_adversarial_loss_generator(loss_type)()
    discriminator_loss = get_adversarial_loss_discriminator(loss_type)()

    # Real data
    data_x, data_y = (
        tf.zeros((dataset_size, image_resolution[0], image_resolution[1])),
        tf.zeros((dataset_size,)),
    )

    # Trainer
    epochs = 2
    metrics = []
    trainer = AdversarialTrainer(
        generator,
        discriminator,
        tf.optimizers.Adam(1e-4),
        tf.optimizers.Adam(1e-4),
        generator_loss,
        discriminator_loss,
        epochs,
        metrics,
        adversarial_logdir,
    )

    # Dataset
    # take only 2 samples to speed up tests
    real_data = (
        tf.data.Dataset.from_tensor_slices(
            (tf.expand_dims(data_x, -1), tf.expand_dims(data_y, -1))
        )
        .take(batch_size)
        .batch(batch_size)
        .prefetch(1)
    )

    # Add noise in the same dataset, just by mapping.
    # The return type of the dataset must be: tuple(tuple(a,b), noise)
    dataset = real_data.map(
        lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, latent_dim)))
    )

    trainer(dataset)
    shutil.rmtree(adversarial_logdir)
