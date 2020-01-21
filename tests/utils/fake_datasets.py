# Copyright 2020 Zuru Tech HK Limited. All Rights Reserved.
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

import tensorflow as tf


def fake_autoencoder_datasest(
    dataset_size=10, image_resolution=(64, 64), channels=3, batch_size=5,
):
    """
    Generate the test dataset for the fake Convolutional Autoencoder model.

    For the Conv Autoencoder the dataset is composed by:
        - inputs: images
        - labels: images

    """
    inputs, labels = (
        tf.zeros((dataset_size, image_resolution[0], image_resolution[1], channels)),
        tf.zeros((dataset_size, image_resolution[0], image_resolution[1], channels)),
    )
    dataset = (
        tf.data.Dataset.from_tensor_slices((inputs, labels))
        .take(dataset_size)
        .batch(batch_size)
        .prefetch(1)
    )

    return dataset
