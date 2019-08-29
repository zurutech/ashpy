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
Test Metrics
"""
import os
import shutil

import json
import tensorflow as tf

from ashpy.metrics import SlicedWasserseinDistance, SSIM_Multiscale, InceptionScore
from ashpy.models.gans import ConvGenerator, ConvDiscriminator
from ashpy.losses.gan import GeneratorBCE, DiscriminatorMinMax
from ashpy.trainers import AdversarialTrainer


def test_metrics(adversarial_logdir: str):
    """
    Test the integration between metrics and trainer
    """
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
        SlicedWasserseinDistance(
            logdir=adversarial_logdir, resolution=image_resolution[0]
        ),
        SSIM_Multiscale(logdir=adversarial_logdir),
        InceptionScore(
            # Fake inception model
            ConvDiscriminator(
                layer_spec_input_res=(299, 299),
                layer_spec_target_res=(7, 7),
                kernel_size=(5, 5),
                initial_filters=16,
                filters_cap=32,
                output_shape=10,
            ),
            logdir=adversarial_logdir,
        ),
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
        adversarial_logdir,
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

    # assert there exists folder for each metric
    for metric in metrics:
        metric_dir = os.path.join(adversarial_logdir, "best", metric.name)
        assert os.path.exists(metric_dir)
        json_path = os.path.join(metric_dir, f"{metric.name}.json")
        assert os.path.exists(json_path)
        with open(json_path, "r") as fp:
            metric_data = json.load(fp)

            # assert the metric data contains the expected keys
            assert metric.name in metric_data
            assert "step" in metric_data

    shutil.rmtree(adversarial_logdir)
