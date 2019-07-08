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

"""Bigan dummy implementation."""

import operator

import tensorflow as tf
from tensorflow import keras  # pylint: disable=no-name-in-module

from ashpy.losses.gan import DiscriminatorMinMax, EncoderBCE, GeneratorBCE
from ashpy.metrics import EncodingAccuracy
from ashpy.trainers import EncoderTrainer


def main():
    """Main train loop and models definition."""

    def real_gen():
        """generator of real values."""
        for _ in tf.range(100):
            yield ((10.0,), (0,))

    num_classes = 1
    latent_dim = 100

    generator = keras.Sequential([keras.layers.Dense(1)])

    left_input = tf.keras.layers.Input(shape=(1,))
    left = tf.keras.layers.Dense(10, activation=tf.nn.elu)(left_input)

    right_input = tf.keras.layers.Input(shape=(latent_dim,))
    right = tf.keras.layers.Dense(10, activation=tf.nn.elu)(right_input)

    net = tf.keras.layers.Concatenate()([left, right])
    out = tf.keras.layers.Dense(1)(net)

    discriminator = tf.keras.Model(inputs=[left_input, right_input], outputs=[out])

    encoder = keras.Sequential([keras.layers.Dense(latent_dim)])
    generator_bce = GeneratorBCE()
    encoder_bce = EncoderBCE()
    minmax = DiscriminatorMinMax()

    epochs = 100
    logdir = "log/adversarial/encoder"

    # Fake pre-trained classifier
    classifier = tf.keras.Sequential(
        [tf.keras.layers.Dense(10), tf.keras.layers.Dense(num_classes)]
    )

    metrics = [
        EncodingAccuracy(
            classifier, model_selection_operator=operator.gt, logdir=logdir
        )
    ]

    trainer = EncoderTrainer(
        generator,
        discriminator,
        encoder,
        tf.optimizers.Adam(1e-4),
        tf.optimizers.Adam(1e-5),
        tf.optimizers.Adam(1e-6),
        generator_bce,
        minmax,
        encoder_bce,
        epochs,
        metrics=metrics,
        logdir=logdir,
    )

    batch_size = 10
    discriminator_input = tf.data.Dataset.from_generator(
        real_gen, (tf.float32, tf.int64), ((1), (1))
    ).batch(batch_size)

    dataset = discriminator_input.map(
        lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, latent_dim)))
    )

    trainer(dataset)


if __name__ == "__main__":
    main()
