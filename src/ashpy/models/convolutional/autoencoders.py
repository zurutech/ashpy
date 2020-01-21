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

"""Collection of Fully Convolutional Autoencoders."""
from ashpy.models.convolutional.decoders import Decoder, FCNNDecoder
from ashpy.models.convolutional.encoders import Encoder, FCNNEncoder
from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["Autoencoder", "FCNNAutoencoder"]


class Autoencoder(keras.Model):  # pylint: disable=no-member
    """
    Primitive Model for all convolutional autoencoders.

    Examples:
        * Direct Usage:

            .. testcode::

                autoencoder = Autoencoder(
                    layer_spec_input_res=(64, 64),
                    layer_spec_target_res=(8, 8),
                    kernel_size=5,
                    initial_filters=32,
                    filters_cap=128,
                    encoding_dimension=100,
                    channels=3,
                )

                encoding, reconstruction = autoencoder(tf.zeros((1, 64, 64, 3)))
                print(encoding.shape)
                print(reconstruction.shape)

            .. testoutput::
                (1, 100)
                (1, 64, 64, 3)

    """

    def __init__(
        self,
        layer_spec_input_res,
        layer_spec_target_res,
        kernel_size,
        initial_filters,
        filters_cap,
        encoding_dimension,
        channels,
    ):
        """
        Instantiate the :py:class:`BaseAutoEncoder`.

        Args:
            layer_spec_input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape of
                the input tensors.
            layer_spec_target_res: (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape
                of tensor desired as output of :func:`_get_layer_spec`.
            kernel_size (int): Kernel used by the convolution layers.
            initial_filters (int): Numbers of filters to used as a base value.
            filters_cap (int): Cap filters to a set amount, in the case of an Encoder is a
                ceil value AKA the max amount of filters.
            encoding_dimension (int): encoding dimension.
            channels (int): Number of channels for the reconstructed image.

        Returns:
            :py:obj:`None`

        """
        super().__init__()
        self._encoder = Encoder(
            layer_spec_input_res,
            layer_spec_target_res,
            kernel_size,
            initial_filters,
            filters_cap,
            encoding_dimension,
        )
        self._decoder = Decoder(
            layer_spec_target_res,
            layer_spec_input_res,
            kernel_size,
            filters_cap,
            initial_filters,
            channels,
        )

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


class FCNNAutoencoder(keras.Model):  # pylint: disable=no-member
    """
    Primitive Model for all fully convolutional autoencoders.

    Examples:
        * Direct Usage:
            .. testcode::

                autoencoder = FCNNAutoencoder(
                    layer_spec_input_res=(64, 64),
                    layer_spec_target_res=(8, 8),
                    kernel_size=5,
                    initial_filters=32,
                    filters_cap=128,
                    encoding_dimension=100,
                    channels=3,
                )

                encoding, reconstruction = autoencoder(tf.zeros((1, 64, 64, 3)))
                print(encoding.shape)
                print(reconstruction.shape)

            .. testoutput::
                (1, 1, 1, 100)
                (1, 64, 64, 3)

    """

    def __init__(
        self,
        layer_spec_input_res,
        layer_spec_target_res,
        kernel_size,
        initial_filters,
        filters_cap,
        encoding_dimension,
        channels,
    ):
        """
        Instantiate the :py:class:`FCNNBaseAutoEncoder`.

        Args:
            layer_spec_input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape of
                the input tensors.
            layer_spec_target_res: (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape
                of tensor desired as output of :func:`_get_layer_spec`.
            kernel_size (int): Kernel used by the convolution layers.
            initial_filters (int): Numbers of filters to used as a base value.
            filters_cap (int): Cap filters to a set amount, in the case of an Encoder is a
                ceil value AKA the max amount of filters.
            encoding_dimension (int): encoding dimension.
            channels (int): Number of channels for the reconstructed image.

        Returns:
            :py:obj:`None`

        """
        super().__init__()
        self._encoder = FCNNEncoder(
            layer_spec_input_res,
            layer_spec_target_res,
            kernel_size,
            initial_filters,
            filters_cap,
            encoding_dimension,
        )
        self._decoder = FCNNDecoder(
            layer_spec_target_res,
            layer_spec_input_res,
            kernel_size,
            filters_cap,
            initial_filters,
            channels,
        )

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
