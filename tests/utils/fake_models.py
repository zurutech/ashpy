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
from ashpy.models.convolutional.autoencoders import Autoencoder


def conv_autoencoder(
    layer_spec_input_res=(64, 64),
    layer_spec_target_res=(4, 4),
    kernel_size=3,
    initial_filters=16,
    filters_cap=64,
    encoding_dimension=50,
    channels=3,
) -> tf.keras.Model:
    """Create a new Convolutinal Autoencoder."""
    autoencoder = Autoencoder(
        layer_spec_input_res=layer_spec_input_res,
        layer_spec_target_res=layer_spec_target_res,
        kernel_size=kernel_size,
        initial_filters=initial_filters,
        filters_cap=filters_cap,
        encoding_dimension=encoding_dimension,
        channels=channels,
    )
    # Encoding, representation = autoencoder(input)
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    _, reconstruction = autoencoder(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=reconstruction)
    return model
