import shutil

import pytest
import tensorflow as tf

from ashpy.metrics import InceptionScore, SlicedWasserseinDistance, SSIM_Multiscale
from ashpy.models.gans import ConvGenerator, ConvDiscriminator
from ashpy.losses.gan import GeneratorBCE, DiscriminatorMinMax
from ashpy.trainers import AdversarialTrainer


def test_losses(logdir: str):
    # test parameters
    image_resolution = (256, 256)
    kernel_size = (5, 5)
    batch_size = 2
    dataset_size = 100
    latent_dim = 100

    # model definition
    generator = ConvGenerator(
        layer_spec_input_res=(8, 8),
        layer_spec_target_res=image_resolution,
        kernel_size=kernel_size,
        initial_filters=32,
        filters_cap=16,
        channels=3,
    )

    discriminator = ConvDiscriminator(
        layer_spec_input_res=image_resolution,
        layer_spec_target_res=(8, 8),
        kernel_size=kernel_size,
        initial_filters=16,
        filters_cap=32,
        output_shape=1,
    )

    # Losses
    generator_loss = GeneratorBCE()
    discriminator_loss = DiscriminatorMinMax()

    # Real data
    data_x, data_y = (
        tf.zeros((dataset_size, image_resolution[0], image_resolution[1], 3)),
        tf.zeros((dataset_size,)),
    )

    # Trainer
    epochs = 2
    metrics = [
        SlicedWasserseinDistance(logdir=logdir, resolution=image_resolution[0]),
        SSIM_Multiscale(logdir=logdir),
    ]
    trainer = AdversarialTrainer(
        generator,
        discriminator,
        tf.optimizers.Adam(1e-4),
        tf.optimizers.Adam(1e-4),
        generator_loss,
        discriminator_loss,
        epochs,
        metrics,
        logdir,
    )

    # Dataset
    # take only 2 samples to speed up tests
    real_data = (
        tf.data.Dataset.from_tensor_slices((data_x, data_y))
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
    shutil.rmtree(logdir)
