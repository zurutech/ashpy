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

import operator

import tensorflow as tf

from ashpy.losses.classifier import ClassifierLoss
from ashpy.metrics import ClassifierMetric
from ashpy.trainers.classifier import ClassifierTrainer


def main():
    """How to use ash to train a classifier, measure the
    performance and perform model selection."""

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        train, validation = tf.keras.datasets.mnist.load_data()

        def process(images, labels):
            data_images = tf.data.Dataset.from_tensor_slices((images)).map(
                lambda x: tf.reshape(x, (28 * 28,))
            )
            data_images = data_images.map(
                lambda x: tf.image.convert_image_dtype(x, tf.float32)
            )
            data_labels = tf.data.Dataset.from_tensor_slices((labels))
            dataset = tf.data.Dataset.zip((data_images, data_labels))
            dataset = dataset.batch(1024 * 1)
            return dataset

        train, validation = (
            process(train[0], train[1]),
            process(validation[0], validation[1]),
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
                tf.keras.layers.Dense(10),
            ]
        )
        optimizer = tf.optimizers.Adam(1e-3)
        loss = ClassifierLoss(tf.losses.SparseCategoricalCrossentropy(from_logits=True))
        logdir = "testlog"
        epochs = 10

        metrics = [
            ClassifierMetric(
                tf.metrics.Accuracy(), model_selection_operator=operator.gt
            ),
            ClassifierMetric(
                tf.metrics.BinaryAccuracy(), model_selection_operator=operator.gt
            ),
        ]

        trainer = ClassifierTrainer(
            model, optimizer, loss, epochs, metrics, logdir=logdir
        )
        trainer(train, validation)


if __name__ == "__main__":
    main()
