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

"""Collection of Decoders (i.e., GANs' Generators) models."""
import tensorflow as tf
from ashpy.models.convolutional.interfaces import Conv2DInterface
from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["Decoder", "FCNNDecoder"]


class Decoder(Conv2DInterface):
    """
    Primitive Model for all decoder (i.e., transpose convolution) based architecture.

    Notes:
        Default to DCGAN Generator architecture.

    Examples:
        * Direct Usage:

            .. testcode::

                dummy_generator = Decoder(
                    layer_spec_input_res=(8, 8),
                    layer_spec_target_res=(64, 64),
                    kernel_size=(5, 5),
                    initial_filters=1024,
                    filters_cap=16,
                    channels=3,
                )

        * Subclassing

            .. testcode::

                class DummyGenerator(Decoder):
                    def call(self, input, training=True):
                        print("Dummy Generator!")
                        return input

                dummy_generator = DummyGenerator(
                    layer_spec_input_res=(8, 8),
                    layer_spec_target_res=(32, 32),
                    kernel_size=(5, 5),
                    initial_filters=1024,
                    filters_cap=16,
                    channels=3,
                )
                dummy_generator(tf.random.normal((1, 100)))

            .. testoutput::

                Dummy Generator!

    """

    def __init__(
        self,
        layer_spec_input_res,
        layer_spec_target_res,
        kernel_size,
        initial_filters,
        filters_cap,
        channels,
        use_dropout=True,
        dropout_prob=0.3,
        non_linearity=keras.layers.LeakyReLU,
    ):
        r"""
        Instantiate the :class:`Decoder`.

        Model Assembly:
            1. :func:`_add_initial_block`: Ingest the :py:obj:`tf.keras.Model`
            inputs and prepare them for :func:`_add_building_block`.

            2. :func:`_add_building_block`: Core of the model, the layers specified
            here get added to the :py:obj:`tf.keras.Model` multiple times consuming the
            hyperparameters generated in the :func:`_get_layer_spec`.

            3. :func:`_add_final_block`: Final block of our :py:obj:`tf.keras.Model`,
            take the model after :func:`_add_building_block` and prepare them for the
            for the final output.

        Args:
            layer_spec_input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape of
                the :func:`_get_layer_spec` input tensors.
            layer_spec_target_res: (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape
                of tensor desired as output of :func:`_get_layer_spec`.
            kernel_size (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Kernel used by the
                convolution layers.
            initial_filters (int): Numbers of filters at the end of the first block.
            filters_cap (int): Cap filters to a set amount, in the case of Decoder is a
                floor value AKA the minimum amount of filters.
            channels (int): Channels of the output images (1 for Grayscale, 3 for RGB).

        Returns:
            :py:obj:`None`

        Raises:
            ValueError: If `filters_cap` > `initial_filters`.

        """
        super().__init__()
        if filters_cap > initial_filters:
            raise ValueError(
                "`filters_cap` > `initial_filters`. "
                "When decoding ``filters_cap`` is a floor value AKA the minimum "
                "amount of filters."
            )

        if isinstance(layer_spec_input_res, int):
            layer_spec_input_res = (layer_spec_input_res, layer_spec_input_res)

        if isinstance(layer_spec_target_res, int):
            layer_spec_target_res = (layer_spec_target_res, layer_spec_target_res)

        filters = self._get_layer_spec(
            initial_filters, filters_cap, layer_spec_input_res, layer_spec_target_res
        )

        # layer specification
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.non_linearity = non_linearity
        self.kernel_size = kernel_size

        # Assembling Model
        self._add_initial_block(initial_filters, layer_spec_input_res)
        for layer_filters in filters:
            self._add_building_block(layer_filters)
        self._add_final_block(channels)

    def _add_initial_block(self, initial_filters, input_res):
        """
        Ingest the :py:obj:`tf.keras.Model` inputs and prepare them for :func:`_add_building_block`.

        Args:
            initial_filters (int): Numbers of filters to used as a base value.
            input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape of the
                :func:`_get_layer_spec` input tensors.

        """
        self.model_layers.extend(
            [
                keras.layers.Dense(initial_filters * input_res[0] * input_res[1]),
                keras.layers.BatchNormalization(),
                self.non_linearity(),
                keras.layers.Reshape((input_res[0], input_res[1], initial_filters)),
            ]
        )

    def _add_building_block(self, filters):
        """
        Construct the core of the :py:obj:`tf.keras.Model`.

        The layers specified here get added to the :py:obj:`tf.keras.Model` multiple times
        consuming the hyperparameters generated in the :func:`_get_layer_spec`.

        Args:
            filters (int): Number of filters to use for this iteration of the Building Block.

        """
        self.model_layers.extend(
            [
                keras.layers.Conv2DTranspose(
                    filters,
                    self.kernel_size,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.BatchNormalization(),
                self.non_linearity(),
            ]
        )

    def _add_final_block(self, channels):
        """
        Prepare results of :func:`_add_building_block` for the for the final output.

        Args:
            channels (int): Channels of the output images (1 for Grayscale, 3 for RGB).

        """
        self.model_layers.append(
            keras.layers.Conv2DTranspose(
                channels,
                self.kernel_size,
                strides=(1, 1),
                padding="same",
                use_bias=False,
                activation=tf.math.tanh,
            )
        )


class FCNNDecoder(Decoder):
    """Fully Convolutional Decoder. Expected input is a feature map.

    Examples:
        * Direct Usage:
            .. testcode::

                dummy_generator = FCNNDecoder(
                    layer_spec_input_res=(8, 8),
                    layer_spec_target_res=(64, 64),
                    kernel_size=(5, 5),
                    initial_filters=1024,
                    filters_cap=16,
                    channels=3,
                )

                print(dummy_generator(tf.zeros((1, 1, 1, 100))).shape)

            .. testoutput::

                (1, 64, 64, 3)

    """

    def __init__(
        self,
        layer_spec_input_res,
        layer_spec_target_res,
        kernel_size,
        initial_filters,
        filters_cap,
        channels,
        use_dropout=True,
        dropout_prob=0.3,
        non_linearity=keras.layers.LeakyReLU,
    ):
        """Build a Fully Convolutional Decoder."""
        self._kernel_size = kernel_size
        super().__init__(
            layer_spec_input_res,
            layer_spec_target_res,
            kernel_size,
            initial_filters,
            filters_cap,
            channels,
            use_dropout=use_dropout,
            dropout_prob=dropout_prob,
            non_linearity=non_linearity,
        )

    def _add_initial_block(self, initial_filters, input_res):
        """
        Ingest the :py:obj:`tf.keras.Model` inputs and prepare them for :func:`_add_building_block`.

        Args:
            initial_filters (int): Numbers of filters to used as a base value.
            input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Shape of the
                :func:`_get_layer_spec` input tensors.

        """
        # Always suppose to have a 1x1xD input vector.
        # GOAL: upsample in order to make it input_res[0], input_res[1], initial_filters
        # Since conv2dtrasponse output is: input size * stride if padding == same
        # and (input size -1) * stride + Kernel size if padding == valid
        # Since input resolution is 1, computing the stride value is
        # not possible (division by zero (input_size-1))
        # hence we have to use padding = same.
        stride = max(*input_res)
        self.model_layers.extend(
            [
                keras.layers.Conv2DTranspose(
                    initial_filters,
                    self._kernel_size,
                    strides=(stride, stride),
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.LeakyReLU(),
            ]
        )
