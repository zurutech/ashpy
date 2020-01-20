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

"""
Pix2Pix HD Implementation.

See: "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs" [1]_

Global Generator + Local Enhancer

.. [1] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs:
    https://arxiv.org/abs/1711.11585

"""
import typing

import tensorflow as tf
from ashpy.layers import InstanceNormalization
from ashpy.models.convolutional.interfaces import Conv2DInterface
from tensorflow import keras

__ALL__ = ["LocalEnhancer", "GlobalGenerator"]


class LocalEnhancer(keras.Model):
    """
    Local Enhancer module of the Pix2PixHD architecture.

    Example:
        .. testcode::

            # instantiate the model
            model = LocalEnhancer()

            # call the model passing inputs
            inputs = tf.ones((1, 512, 512, 3))
            output = model(inputs)

            # the output shape is
            # the same as the input shape
            print(output.shape)

        .. testoutput::

            (1, 512, 512, 3)

    """

    def __init__(
        self,
        input_res: int = 512,
        min_res: int = 64,
        initial_filters: int = 64,
        filters_cap: int = 512,
        channels: int = 3,
        normalization_layer: typing.Type[keras.layers.Layer] = InstanceNormalization,
        non_linearity: typing.Type[keras.layers.Layer] = keras.layers.ReLU,
        num_resnet_blocks_global: int = 9,
        num_resnet_blocks_local: int = 3,
        kernel_size_resnet: int = 3,
        kernel_size_front_back: int = 7,
        num_internal_resnet_blocks: int = 2,
    ):
        """
        Build the LocalEnhancer module of the Pix2PixHD architecture.

        See High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs [2]_
        for more details.

        Args:
            input_res (int): input resolution.
            min_res (int): minimum resolution reached by the global generator.
            initial_filters (int): number of initial filters.
            filters_cap (int): maximum number of filters.
            channels (int): number of channels.
            normalization_layer (:class:`tf.keras.layers.Layer`): layer of normalization
            (e.g. Instance Normalization or BatchNormalization or LayerNormalization).
            non_linearity (:class:`tf.keras.layers.Layer`): non linearity used in Pix2Pix HD.
            num_resnet_blocks_global (int): number of residual blocks used
                in the global generator.
            num_resnet_blocks_local (int): number of residual blocks used in the local generator.
            kernel_size_resnet (int): kernel size used in resnets.
            kernel_size_front_back (int): kernel size used for the front and back convolution.
            num_internal_resnet_blocks (int): number of internal blocks of the resnet.

        .. [2] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
            https://arxiv.org/abs/1711.11585

        """
        super(LocalEnhancer, self).__init__()
        self.global_generator = GlobalGenerator(
            int(input_res / 2),
            min_res=min_res,
            initial_filters=initial_filters,
            filters_cap=filters_cap,
            channels=channels,
            normalization_layer=normalization_layer,
            non_linearity=non_linearity,
            num_resnet_blocks=num_resnet_blocks_global,
            kernel_size_front_back=kernel_size_front_back,
            kernel_size_resnet=kernel_size_resnet,
            num_internal_resnet_blocks=num_internal_resnet_blocks,
        )
        self.downsample = keras.layers.AvgPool2D(3, strides=2)
        self.downsample_block = [  # add padding ?
            tf.keras.layers.Conv2D(
                initial_filters,
                kernel_size=kernel_size_front_back,
                strides=1,
                padding="same",
            ),
            normalization_layer(),
            non_linearity(),
            tf.keras.layers.Conv2D(
                initial_filters * 2,
                kernel_size=kernel_size_resnet,
                strides=2,
                padding="same",
            ),
            normalization_layer(),
            non_linearity(),
        ]

        # resnet blocks
        self.resnet_blocks = [
            ResNetBlock(
                initial_filters * 2,
                non_linearity=non_linearity,
                normalization_layer=normalization_layer,
                kernel_size=kernel_size_resnet,
                num_blocks=num_internal_resnet_blocks,
            )
            for _ in range(num_resnet_blocks_local)
        ]

        # upsample
        self.upsample_block = [
            keras.layers.Conv2DTranspose(
                initial_filters * 2,
                kernel_size=kernel_size_resnet,
                strides=2,
                padding="same",
            ),
            normalization_layer(),
            non_linearity(),
        ]

        # final convolution
        self.final_layer = keras.layers.Conv2D(
            channels,
            kernel_size=kernel_size_front_back,
            activation=tf.nn.tanh,
            padding="same",
        )

    # un-comment in order to export the model
    # @tf.function(
    #    input_signature=[tf.TensorSpec(shape=[None, 512, 512, 1], dtype=tf.float32)]
    # )
    def call(self, inputs, training=False):
        """
        Call the LocalEnhancer model.

        Args:
            inputs (:py:class:`tf.Tensor`): Input Tensors.
            training (bool): Whether it is training phase or not.

        Returns:
            (:py:class:`tf.Tensor`): Image of size (input_res, input_res, channels)
                as specified in the init call.

        """
        downsampled = self.downsample(inputs)

        # call the global generator
        _, global_generator_features = self.global_generator(downsampled)

        # first downsample
        x = inputs
        for layer in self.downsample_block:
            if isinstance(
                layer, (keras.layers.BatchNormalization, keras.layers.Dropout)
            ):
                x = layer(x, trainig=training)
            else:
                x = layer(x)

        # then add the downsampled and the output of the global generator
        x = x + global_generator_features

        for layer in self.resnet_blocks:
            x = layer(x, training=training)

        # upsample
        for layer in self.upsample_block:
            if isinstance(
                layer, (keras.layers.BatchNormalization, keras.layers.Dropout)
            ):
                x = layer(x, training=training)
            else:
                x = layer(x)

        # final block
        x = self.final_layer(x)

        return x


class ResNetBlock(keras.Model):
    """
    ResNet Blocks.

    The input filters is the same as the output filters.
    """

    def __init__(
        self,
        filters: int,
        normalization_layer: typing.Type[keras.layers.Layer] = InstanceNormalization,
        non_linearity: typing.Type[keras.layers.Layer] = keras.layers.ReLU,
        kernel_size: int = 3,
        num_blocks: int = 2,
    ):
        """
        Build the ResNet block composed by num_blocks.

        Each block is composed by
         - Conv2D with strides 1 and padding "same"
         - Normalization Layer
         - Non Linearity

        The final result is the output of the ResNet + input.

        Args:
            filters (int): initial filters (same as the output filters).
            normalization_layer (:class:`tf.keras.layers.Layer`): layer of normalization
                used by the residual block.
            non_linearity (:class:`tf.keras.layers.Layer`): non linearity used
                in the resnet block.
            kernel_size (int): kernel size used in the resnet block.
            num_blocks (int): number of blocks, each block is composed by conv,
                normalization and non linearity.

        """
        super(ResNetBlock, self).__init__()
        self.model_layers = []
        for _ in range(num_blocks):
            self.model_layers.extend(
                [
                    keras.layers.Conv2D(
                        filters, kernel_size=kernel_size, padding="same", strides=1
                    ),
                    normalization_layer(),
                    non_linearity(),
                ]
            )

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: input tensor.
            training: whether is training or not.

        Returns:
            A Tensor of the same shape as the inputs.
            The input passed through num_blocks blocks.

        """
        out = inputs
        for layer in self.model_layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                out = layer(out, training=training)
            else:
                out = layer(out)
        return inputs + out


class GlobalGenerator(Conv2DInterface):
    """
    Global Generator from pix2pixHD paper.

     - G1^F: Convolutional frontend (downsampling)
     - G1^R: ResNet Block
     - G1^B: Convolutional backend (upsampling)
    """

    def __init__(
        self,
        input_res: int = 512,
        min_res: int = 64,
        initial_filters: int = 64,
        filters_cap: int = 512,
        channels: int = 3,
        normalization_layer: typing.Type[keras.layers.Layer] = InstanceNormalization,
        non_linearity: typing.Type[keras.layers.Layer] = keras.layers.ReLU,
        num_resnet_blocks: int = 9,
        kernel_size_resnet: int = 3,
        kernel_size_front_back: int = 7,
        num_internal_resnet_blocks: int = 2,
    ):
        """
        Global Generator from Pix2PixHD.

        Args:
            input_res (int): Input Resolution.
            min_res (int): Minimum resolution reached by the downsampling.
            initial_filters (int): number of initial filters.
            filters_cap (int): maximum number of filters.
            channels (int): output channels.
            normalization_layer (:class:`tf.keras.layers.Layer`): normalization layer used
                by the global generator, can be Instance Norm, Layer Norm, Batch Norm.
            non_linearity (:class:`tf.keras.layers.Layer`): non linearity
                used in the global generator.
            num_resnet_blocks (int): number of resnet blocks.
            kernel_size_resnet (int): kernel size used in resnets conv layers.
            kernel_size_front_back (int): kernel size used by the convolutional
                frontend and backend.
            num_internal_resnet_blocks (int): number of blocks used by internal resnet.

        """
        super().__init__()
        self.first_block = [
            tf.keras.layers.Conv2D(
                initial_filters,
                kernel_size=kernel_size_front_back,
                strides=1,
                padding="same",
            ),
            normalization_layer(),
            non_linearity(),
        ]

        self.downsample_blocks = []

        # Downsample
        layer_spec = self._get_layer_spec(
            initial_filters * 2, filters_cap, input_res, min_res
        )
        for filters in layer_spec:
            self.downsample_blocks.extend(
                [
                    tf.keras.layers.Conv2D(
                        filters,
                        kernel_size=kernel_size_resnet,
                        strides=2,
                        padding="same",
                    ),
                    normalization_layer(),
                    non_linearity(),
                ]
            )

        # ResNet Block
        self.resnet_blocks = []
        for _ in range(num_resnet_blocks):
            self.resnet_blocks.append(
                ResNetBlock(
                    filters,
                    non_linearity=non_linearity,
                    normalization_layer=normalization_layer,
                    kernel_size=kernel_size_resnet,
                    num_blocks=num_internal_resnet_blocks,
                )
            )

        # upsample
        self.upsample_blocks = []
        layer_spec = self._get_layer_spec(filters, initial_filters, min_res, input_res)
        for filters in layer_spec:
            self.upsample_blocks.extend(
                [
                    tf.keras.layers.Conv2DTranspose(
                        filters,
                        kernel_size=kernel_size_resnet,
                        strides=2,
                        padding="same",
                    ),
                    normalization_layer(),
                    non_linearity(),
                ]
            )

        self.last_layer = keras.layers.Conv2D(
            channels,
            kernel_size=kernel_size_front_back,
            strides=1,
            activation=keras.activations.tanh,
            padding="same",
        )

        self.model_layers = []
        self.model_layers.extend(self.first_block)
        self.model_layers.extend(self.downsample_blocks)
        self.model_layers.extend(self.resnet_blocks)
        self.model_layers.extend(self.upsample_blocks)
        self.model_layers.append(self.last_layer)

    def call(self, inputs, training=True):
        """
        Call of the Pix2Pix HD model.

        Args:
            inputs: input tensor(s).
            training: If True training phase.

        Returns:
            :py:class:`Tuple`: Generated images.

        """
        out = inputs
        prev = inputs
        for layer in self.model_layers:
            prev = out
            if isinstance(
                layer,
                (ResNetBlock, keras.layers.BatchNormalization, keras.layers.Dropout),
            ):
                out = layer(prev, training=training)
            else:
                out = layer(prev)

        return out, prev
