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

"""Fake training loop to simplify training in tests."""
import operator

import ashpy
import tensorflow as tf
from ashpy.losses import DiscriminatorMinMax, GeneratorBCE
from ashpy.models.gans import ConvDiscriminator, ConvGenerator
from ashpy.trainers import AdversarialTrainer

from tests.utils.fake_datasets import (
    fake_adversarial_dataset,
    fake_autoencoder_datasest,
)
from tests.utils.fake_models import basic_dcgan, conv_autoencoder

__ALL__ = ["fake_classifier_training_loop", "fake_adversarial_training_loop"]


def fake_classifier_training_loop(
    # Trainer
    logdir: str = "testlog",
    optimizer=tf.optimizers.Adam(1e-4),
    metrics=[ashpy.metrics.ClassifierLoss(model_selection_operator=operator.lt)],
    epochs=2,
    # Dataset
    dataset_size=10,
    image_resolution=(64, 64),
    batch_size=5,
    # Model: Autoencoder
    layer_spec_input_res=(64, 64),
    layer_spec_target_res=(4, 4),
    kernel_size=3,
    initial_filters=16,
    filters_cap=64,
    encoding_dimension=50,
    channels=3,
    # Call parameters
    measure_performance_freq=10,
):
    """Fake Classifier training loop implementation using an autoencoder as a base model."""
    # Model
    model = conv_autoencoder(
        layer_spec_input_res,
        layer_spec_target_res,
        kernel_size,
        initial_filters,
        filters_cap,
        encoding_dimension,
        channels,
    )

    # Dataset
    dataset = fake_autoencoder_datasest(
        dataset_size, image_resolution, channels, batch_size
    )

    # Loss
    reconstruction_error = ashpy.losses.ClassifierLoss(
        tf.keras.losses.MeanSquaredError()
    )

    # Trainer
    trainer = ashpy.trainers.ClassifierTrainer(
        model=model,
        optimizer=optimizer,
        loss=reconstruction_error,
        logdir=str(logdir),
        epochs=epochs,
        metrics=metrics,
    )

    trainer(dataset, dataset, measure_performance_freq=measure_performance_freq)
    return 1, trainer


def fake_adversarial_training_loop(
    logdir: str = "testlog",
    kernel_size=(5, 5),
    metrics=None,
    callbacks=None,
    epochs=2,
    dataset_size=2,
    batch_size=2,
    generator_loss=GeneratorBCE(),
    discriminator_loss=DiscriminatorMinMax(),
    image_resolution=(28, 28),
    layer_spec_input_res=(7, 7),
    layer_spec_target_res=(7, 7),
    channels=1,
    output_shape=1,
    latent_dim=100,
    # Call parameters
    measure_performance_freq=10,
    # Models from outside
    generator=None,
    discriminator=None,
):
    """Fake training loop implementation."""
    # test parameters
    if callbacks is None:
        callbacks = []
    if metrics is None:
        metrics = []

    # Model definition
    models = basic_dcgan(
        image_resolution=image_resolution,
        layer_spec_input_res=layer_spec_input_res,
        layer_spec_target_res=layer_spec_target_res,
        kernel_size=kernel_size,
        channels=channels,
        output_shape=output_shape,
    )
    if not generator:
        generator = models[0]
    if not discriminator:
        discriminator = models[1]

    # Trainer
    trainer = AdversarialTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=tf.optimizers.Adam(1e-4),
        discriminator_optimizer=tf.optimizers.Adam(1e-4),
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        epochs=epochs,
        metrics=metrics,
        callbacks=callbacks,
        logdir=logdir,
    )

    dataset = fake_adversarial_dataset(
        image_resolution=image_resolution,
        epochs=epochs,
        dataset_size=dataset_size,
        batch_size=batch_size,
        latent_dim=latent_dim,
        channels=channels,
    )

    trainer(dataset, measure_performance_freq=measure_performance_freq)
    return 1, trainer
