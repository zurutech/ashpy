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

"""UNET implementations."""
import typing

import tensorflow as tf
from ashpy.layers import Attention, InstanceNormalization
from ashpy.models.convolutional.interfaces import Conv2DInterface
from tensorflow import keras

__ALL__ = ["UNet", "SUNet", "FUNet"]


class UNet(Conv2DInterface):
    """
    UNet Architecture.

    Architecture similar to the one found in "Image-to-Image Translation
    with Conditional Adversarial Nets" [1]_.

    Originally proposed in "U-Net: Convolutional Networks for Biomedical Image Segmentation" [2]_.

    Examples:
        * Direct Usage:

            .. testcode::

                x = tf.ones((1, 512, 512, 3))
                u_net = UNet(input_res = 512,
                             min_res=4,
                             kernel_size=4,
                             initial_filters=64,
                             filters_cap=512,
                             channels=3)
                y = u_net(x)
                print(y.shape)
                print(len(u_net.trainable_variables)>0)

            .. testoutput::

                (1, 512, 512, 3)
                True

    .. [1] Image-to-Image Translation with Conditional Adversarial Nets -
        https://arxiv.org/abs/1611.07004
    .. [2] U-Net: Convolutional Networks for Biomedical Image Segmentation -
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        input_res: int,
        min_res: int,
        kernel_size: int,
        initial_filters: int,
        filters_cap: int,
        channels: int,
        use_dropout_encoder: bool = True,
        use_dropout_decoder: bool = True,
        dropout_prob: float = 0.3,
        encoder_non_linearity: typing.Type[keras.layers.Layer] = keras.layers.LeakyReLU,
        decoder_non_linearity: typing.Type[keras.layers.Layer] = keras.layers.ReLU,
        normalization_layer: typing.Type[keras.layers.Layer] = InstanceNormalization,
        last_activation: keras.activations = keras.activations.tanh,
        use_attention: bool = False,
    ):
        """
        Initialize the UNet.

        Args:
            input_res: input resolution.
            min_res: minimum resolution reached after decode.
            kernel_size: kernel size used in the network.
            initial_filters: number of filter of the initial convolution.
            filters_cap: maximum number of filters.
            channels: number of output channels.
            use_dropout_encoder: whether to use dropout in the encoder module.
            use_dropout_decoder: whether to use dropout in the decoder module.
            dropout_prob: probability of dropout.
            encoder_non_linearity: non linearity of encoder.
            decoder_non_linearity: non linearity of decoder.
            last_activation: last activation function, tanh or softmax (for semantic images).
            use_attention: whether to use attention.

        """
        super().__init__()

        # layer specification
        self.use_dropout_encoder = use_dropout_encoder
        self.use_dropout_decoder = use_dropout_decoder
        self.dropout_probability = dropout_prob
        self.encoder_non_linearity = encoder_non_linearity
        self.decoder_non_linearity = decoder_non_linearity
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        self.normalization = normalization_layer

        # encoder layers is a list of list, each list is a "block",
        # this makes easy the creation of decoder
        self.encoder_layers = []
        self.decoder_layers = []
        self.concat_layers = []

        # ########### Encoder creation
        encoder_layers_spec = self._get_layer_spec(
            initial_filters, filters_cap, input_res, min_res
        )
        # from generator to list
        encoder_layers_spec = [x for x in encoder_layers_spec]

        decoder_layer_spec = []
        for i, filters in enumerate(encoder_layers_spec):
            decoder_layer_spec.insert(0, filters)
            block = self.get_encoder_block(
                filters,
                use_bn=(i not in (0, len(encoder_layers_spec) - 1)),
                use_attention=i == 2,
            )
            self.encoder_layers.append(block)

        # ############## Decoder creation
        decoder_layer_spec = decoder_layer_spec[1:]

        for i, filters in enumerate(decoder_layer_spec):
            self.concat_layers.append(keras.layers.Concatenate())
            block = self.get_decoder_block(
                filters, use_dropout=(i < 3), use_attention=i == 5
            )
            self.decoder_layers.append(block)

        # final layer
        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.final_layer = keras.layers.Conv2DTranspose(
            channels,
            self.kernel_size,
            strides=(2, 2),
            padding="same",
            activation=last_activation,
            kernel_initializer=initializer,
        )

    def _get_block(
        self,
        filters,
        conv_layer=None,
        use_bn=True,
        use_dropout=False,
        non_linearity=keras.layers.LeakyReLU,
        use_attention=False,
    ):
        initializer = tf.random_normal_initializer(0.0, 0.02)
        # Conv2D
        block = [
            conv_layer(
                filters,
                self.kernel_size,
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=initializer,
            )
        ]

        # Batch normalization
        if use_bn:
            block.append(self.normalization())

        # dropout
        if use_dropout:
            block.append(keras.layers.Dropout(self.dropout_probability))

        # Non linearity
        block.append(non_linearity())

        # attention
        if use_attention:
            block.append(Attention(filters))

        return block

    def get_encoder_block(self, filters, use_bn=True, use_attention=False):
        """
        Return a block to be used in the encoder part of the UNET.

        Args:
            filters: number of filters.
            use_bn: whether to use batch normalization.
            use_attention: whether to use attention.

        Returns:
            A block to be used in the encoder part.

        """
        return self._get_block(
            filters,
            conv_layer=keras.layers.Conv2D,
            use_bn=use_bn,
            use_dropout=self.use_dropout_encoder,
            non_linearity=self.encoder_non_linearity,
            use_attention=use_attention and self.use_attention,
        )

    def get_decoder_block(
        self, filters, use_bn=True, use_dropout=False, use_attention=False
    ):
        """
        Return a block to be used in the decoder part of the UNET.

        Args:
            filters: number of filters
            use_bn: whether to use batch normalization
            use_dropout: whether to use dropout
            use_attention: whether to use attention

        Returns:
            A block to be used in the decoder part

        """
        return self._get_block(
            filters,
            conv_layer=keras.layers.Conv2DTranspose,
            use_bn=use_bn,
            use_dropout=self.use_dropout_decoder and use_dropout,
            non_linearity=self.decoder_non_linearity,
            use_attention=use_attention and self.use_attention,
        )

    # @tf.function(
    #    input_signature=[tf.TensorSpec(shape=[None, 512, 512, 1], dtype=tf.float32)]
    # )
    def call(self, inputs, training=False):
        """Forward pass of the UNet model."""
        encoder_layer_eval = []
        x = inputs

        for block in self.encoder_layers:
            for layer in block:
                if isinstance(
                    layer, (keras.layers.BatchNormalization, keras.layers.Dropout)
                ):
                    x = layer(x, training=training)
                else:
                    x = layer(x)

            encoder_layer_eval.append(x)

        encoder_layer_eval = encoder_layer_eval[:-1]

        for i, block in enumerate(self.decoder_layers):
            for layer in block:
                if isinstance(
                    layer, (keras.layers.BatchNormalization, keras.layers.Dropout)
                ):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            x = self.concat_layers[i]([x, encoder_layer_eval[-1 - i]])

        x = self.final_layer(x)

        return x


class SUNet(UNet):
    """Semantic UNet."""

    def __init__(
        self,
        input_res,
        min_res,
        kernel_size,
        initial_filters,
        filters_cap,
        channels,  # number of classes
        use_dropout_encoder=True,
        use_dropout_decoder=True,
        dropout_prob=0.3,
        encoder_non_linearity=keras.layers.LeakyReLU,
        decoder_non_linearity=keras.layers.ReLU,
        use_attention=False,
    ):
        """Build the Semantic UNet model."""
        super().__init__(
            input_res,
            min_res,
            kernel_size,
            initial_filters,
            filters_cap,
            channels,
            use_dropout_encoder,
            use_dropout_decoder,
            dropout_prob,
            encoder_non_linearity,
            decoder_non_linearity,
            last_activation=keras.activations.softmax,
            use_attention=use_attention,
        )


def FUNet(
    input_res,
    min_res,
    kernel_size,
    initial_filters,
    filters_cap,
    channels,
    input_channels=3,
    use_dropout_encoder=True,
    use_dropout_decoder=True,
    dropout_prob=0.3,
    encoder_non_linearity=keras.layers.LeakyReLU,
    decoder_non_linearity=keras.layers.ReLU,
    last_activation=keras.activations.tanh,  # tanh or softmax (for semantic images)
    use_attention=False,
):
    """Functional UNET Implementation."""
    # ########### Encoder creation
    encoder_layers_spec = Conv2DInterface._get_layer_spec(
        initial_filters, filters_cap, input_res, min_res
    )
    encoder_layers_spec = [x for x in encoder_layers_spec]
    normalization = InstanceNormalization

    def get_block(
        kernel_size,
        filters,
        conv_layer,
        use_bn,
        use_dropout,
        non_linearity,
        use_attention,
        dropout_probability,
    ):
        initializer = tf.random_normal_initializer(0.0, 0.02)
        # Conv2D
        block = [
            conv_layer(
                filters,
                kernel_size,
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=initializer,
            )
        ]

        # Batch normalization
        if use_bn:
            block.append(normalization())

        # dropout
        if use_dropout:
            block.append(keras.layers.Dropout(dropout_probability))

        # Non linearity
        block.append(non_linearity())

        # attention
        if use_attention:
            block.append(Attention(filters))

        return block

    decoder_layer_spec = []
    encoder_layers = []
    concat_layers = []
    decoder_layers = []
    for i, filters in enumerate(encoder_layers_spec):
        decoder_layer_spec.insert(0, filters)
        block = get_block(
            kernel_size,
            filters,
            conv_layer=keras.layers.Conv2D,
            use_bn=(i not in (0, len(encoder_layers_spec) - 1)),
            use_dropout=use_dropout_encoder,
            non_linearity=encoder_non_linearity,
            use_attention=(i == 2 and use_attention),
            dropout_probability=dropout_prob,
        )
        encoder_layers.append(block)

    # ############## Decoder creation
    decoder_layer_spec = decoder_layer_spec[1:]

    for i, filters in enumerate(decoder_layer_spec):
        concat_layers.append(keras.layers.Concatenate())
        block = get_block(
            kernel_size,
            filters,
            conv_layer=keras.layers.Conv2DTranspose,
            use_bn=(i != 0),
            use_dropout=(i < 3) and use_dropout_decoder,
            non_linearity=decoder_non_linearity,
            use_attention=(i == 5 and use_attention),
            dropout_probability=dropout_prob,
        )
        decoder_layers.append(block)

    # final layer
    initializer = tf.random_normal_initializer(0.0, 0.02)
    final_layer = keras.layers.Conv2DTranspose(
        channels,
        kernel_size,
        strides=(2, 2),
        padding="same",
        activation=last_activation,
        kernel_initializer=initializer,
    )
    inputs = tf.keras.layers.Input(shape=[input_res, input_res, input_channels])
    x = inputs
    skips = []

    for block in encoder_layers:
        for layer in block:
            x = layer(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for block, skip in zip(decoder_layers, skips):
        for layer in block:
            x = layer(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = final_layer(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
