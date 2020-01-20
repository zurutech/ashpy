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

"""Collection of Encoders (i.e., GANs' Discriminators) models."""
from typing import Tuple, Type, Union

from ashpy.models.convolutional.interfaces import Conv2DInterface
from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["Encoder", "FCNNEncoder"]


class Encoder(Conv2DInterface):
    """
    Primitive Model for all encoder (i.e., convolution) based architecture.

    Notes:
        Default to DCGAN Discriminator architecture.

    Examples:
        * Direct Usage:

            .. testcode::

                dummy_generator = Encoder(
                    layer_spec_input_res=(64, 64),
                    layer_spec_target_res=(8, 8),
                    kernel_size=5,
                    initial_filters=4,
                    filters_cap=128,
                    output_shape=1,
                )

        * Subclassing

            .. testcode::

                class DummyDiscriminator(Encoder):
                    def call(self, inputs, training=True):
                        print("Dummy Discriminator!")
                        # build the model using
                        # self._layers and inputs
                        return inputs

                dummy_discriminator = DummyDiscriminator(
                    layer_spec_input_res=(64, 64),
                    layer_spec_target_res=(8, 8),
                    kernel_size=5,
                    initial_filters=16,
                    filters_cap=128,
                    output_shape=1,
                )
                dummy_discriminator(tf.zeros((1,28,28,3)))

            .. testoutput::

                Dummy Discriminator!

    """

    def __init__(
        self,
        layer_spec_input_res: Union[int, Tuple[int, int]],
        layer_spec_target_res: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
        initial_filters: int,
        filters_cap: int,
        output_shape: int,
        use_dropout: bool = True,
        dropout_prob: float = 0.3,
        non_linearity: Type[keras.layers.Activation] = keras.layers.LeakyReLU,
    ):
        """
        Instantiate the :py:class:`Decoder`.

        Args:
            layer_spec_input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape of
                the input tensors.
            layer_spec_target_res: (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape
                of tensor desired as output of :func:`_get_layer_spec`.
            kernel_size (int): Kernel used by the convolution layers.
            initial_filters (int): Numbers of filters to used as a base value.
            filters_cap (int): Cap filters to a set amount, in the case of an Encoder is a
                ceil value AKA the max amount of filters.
            output_shape (int): Amount of units of the last :py:obj:`tf.keras.layers.Dense`.

        Returns:
            :py:obj:`None`

        Raises:
            ValueError: If `filters_cap` < `initial_filters`

        """
        super().__init__()
        if filters_cap < initial_filters:
            raise ValueError(
                "`filters_cap` < `initial_filters`. "
                "When decoding ``filters_cap`` is a ceil value AKA the maximum "
                "amount of filters."
            )
        filters = self._get_layer_spec(
            initial_filters, filters_cap, layer_spec_input_res, layer_spec_target_res
        )

        self.model_layers = []

        # layer specification
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.non_linearity = non_linearity
        self.kernel_size = kernel_size

        # Assembling Model
        for layer_filters in filters:
            self._add_building_block(layer_filters)
        self._add_final_block(output_shape)

    def _add_building_block(self, filters):
        """
        Construct the core of the :py:obj:`tf.keras.Model`.

        The layers specified here get added to the :py:obj:`tf.keras.Model` multiple times
        consuming the hyper-parameters generated in the :func:`_get_layer_spec`.

        Args:
            filters (int): Number of filters to use for this iteration of the Building Block.

        """
        self.model_layers.extend(
            [
                keras.layers.Conv2D(
                    filters, self.kernel_size, strides=(2, 2), padding="same"
                ),
                self.non_linearity(),
            ]
        )
        if self.use_dropout:
            self.model_layers.append(keras.layers.Dropout(self.dropout_prob))

    def _add_final_block(self, output_shape):
        """
        Prepare the results of :func:`_add_building_block` for the final output.

        Args:
            output_shape (int): Amount of units of the last :py:obj:`tf.keras.layers.Dense`

        """
        self.model_layers.extend(
            [keras.layers.Flatten(), keras.layers.Dense(output_shape)]
        )


class FCNNEncoder(Encoder):
    """Fully Convolutional Encoder.

    Output a 1x1xencoding_size vector.
    The output neurons are linear.

    Examples:
        * Direct Usage:

            .. testcode::

                dummy_generator = FCNNEncoder(
                    layer_spec_input_res=(64, 64),
                    layer_spec_target_res=(8, 8),
                    kernel_size=5,
                    initial_filters=4,
                    filters_cap=128,
                    encoding_dimension=100,
                )
                print(dummy_generator(tf.zeros((1, 64, 64, 3))).shape)

            .. testoutput::

                (1, 1, 1, 100)

    """

    def __init__(
        self,
        layer_spec_input_res,
        layer_spec_target_res,
        kernel_size,
        initial_filters,
        filters_cap,
        encoding_dimension,
    ):
        """
        Instantiate the :py:class:`FCNNDecoder`.

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

        Returns:
            :py:obj:`None`

        Raises:
            ValueError: If `filters_cap` < `initial_filters`

        """
        self._layer_spec_target_res = layer_spec_target_res
        self._encoding_dimension = encoding_dimension
        super().__init__(
            layer_spec_input_res,
            layer_spec_target_res,
            kernel_size,
            initial_filters,
            filters_cap,
            0,
        )

    def _add_final_block(self, output_shape):
        self.model_layers.append(
            keras.layers.Conv2D(
                self._encoding_dimension,
                self._layer_spec_target_res,
                strides=(1, 1),
                padding="valid",
            )
        )
