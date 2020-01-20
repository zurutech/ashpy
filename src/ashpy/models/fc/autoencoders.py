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

"""Collection of Fully Connected Autoencoders."""
from ashpy.models.fc.decoders import Decoder
from ashpy.models.fc.encoders import Encoder
from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["Autoencoder"]


class Autoencoder(keras.Model):  # pylint: disable=no-member
    """
    Primitive Model for all fully connected autoencoders.

    Examples:
        * Direct Usage:
            .. testcode::

                autoencoder = Autoencoder(
                    hidden_units=[256,128,64],
                    encoding_dimension=100,
                    output_shape=55
                )

                encoding, reconstruction = autoencoder(tf.zeros((1, 55)))
                print(encoding.shape)
                print(reconstruction.shape)

            .. testoutput::
                (1, 100)
                (1, 55)

    """

    def __init__(self, hidden_units, encoding_dimension, output_shape):
        """
        Instantiate the :py:class:`Decoder`.

        Args:
            hidden_units (:obj:`tuple` of :obj:`int`): Number of units per hidden layer.
            encoding_dimension (int): encoding dimension.
            output_shape (int): output shape, usual equal to the input shape.

        Returns:
            :py:obj:`None`

        """
        super().__init__()

        self._encoder = Encoder(hidden_units, encoding_dimension)
        self._decoder = Decoder(hidden_units[::-1], output_shape)

    def call(self, inputs, training=True):
        """
        Execute the model on input data.

        Args:
            inputs (:py:class:`tf.Tensor`): Input tensors.
            training (:obj:`bool`): Training flag.

        Returns:
            (encoding, reconstruction): Pair of tensors.

        """
        encoding = self._encoder(inputs, training)
        reconstruction = self._decoder(encoding, training)
        return encoding, reconstruction
