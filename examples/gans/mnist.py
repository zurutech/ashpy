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

"""Adversarial trainer example."""

import tensorflow as tf
from tensorflow import keras  # pylint: disable=no-name-in-module

from ashpy.losses import DiscriminatorMinMax, GeneratorBCE
from ashpy.metrics import InceptionScore
from ashpy.models.gans import ConvDiscriminator, ConvGenerator
from ashpy.trainers import AdversarialTrainer


def main():
    """Adversarial trainer example."""
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        generator = ConvGenerator(
            layer_spec_input_res=(7, 7),
            layer_spec_target_res=(28, 28),
            kernel_size=(5, 5),
            initial_filters=256,
            filters_cap=16,
            channels=1,
        )

        discriminator = ConvDiscriminator(
            layer_spec_input_res=(28, 28),
            layer_spec_target_res=(7, 7),
            kernel_size=(5, 5),
            initial_filters=32,
            filters_cap=128,
            output_shape=1,
        )

        # Losses
        generator_bce = GeneratorBCE()
        minmax = DiscriminatorMinMax()

        # Trainer
        logdir = "log/adversarial"

        # InceptionScore: keep commented until the issues
        # https://github.com/tensorflow/tensorflow/issues/28599
        # https://github.com/tensorflow/hub/issues/295
        # Haven't been solved and merged into tf2

        metrics = [
            # InceptionScore(
            #    InceptionScore.get_or_train_inception(
            #        mnist_dataset,
            #        "mnist",
            #        num_classes=10,
            #        epochs=1,
            #        fine_tuning=False,
            #        logdir=logdir,
            #    ),
            #    model_selection_operator=operator.gt,
            #    logdir=logdir,
            # )
        ]

        epochs = 50
        trainer = AdversarialTrainer(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=tf.optimizers.Adam(1e-4),
            discriminator_optimizer=tf.optimizers.Adam(1e-4),
            generator_loss=generator_bce,
            discriminator_loss=minmax,
            epochs=epochs,
            metrics=metrics,
            logdir=logdir,
        )

        batch_size = 512

        # Real data
        mnist_x, mnist_y = keras.datasets.mnist.load_data()[0]

        def iterator():
            """Define an iterator in order to do not load in memory all the dataset."""
            for image, label in zip(mnist_x, mnist_y):
                yield tf.image.convert_image_dtype(
                    tf.expand_dims(image, -1), tf.float32
                ), tf.expand_dims(label, -1)

        real_data = (
            tf.data.Dataset.from_generator(
                iterator, (tf.float32, tf.int64), ((28, 28, 1), (1,))
            )
            .batch(batch_size)
            .prefetch(1)
        )

        # Add noise in the same dataset, just by mapping.
        # The return type of the dataset must be: tuple(tuple(a,b), noise)
        dataset = real_data.map(
            lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, 100)))
        )

        trainer(dataset)


if __name__ == "__main__":
    main()
