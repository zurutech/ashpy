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

"""Instance Noramalization Layer implementation."""

from typing import Tuple

import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    r"""
    Instance Normalization Layer (used by Pix2Pix [1]_ and Pix2PixHD [2]_ ).

    Basically it's a normalization done at instance level.
    The implementation follows the basic implementation of the Batch Normalization Layer.

    * Direct Usage:

            .. testcode::

                x = tf.ones((1, 10, 10, 64))

                # instantiate attention layer as model.
                normalization = InstanceNormalization()

                # evaluate passing x.
                output = normalization(x)

                # the output shape is.
                # the same as the input shape.
                print(output.shape)

        * Inside a Model:

            .. testcode::

                def MyModel():
                    inputs = tf.keras.layers.Input(shape=[None, None, 64])
                    normalization = InstanceNormalization()
                    return tf.keras.Model(inputs=inputs, outputs=normalization(inputs))

                x = tf.ones((1, 10, 10, 64))
                model = MyModel()
                output = model(x)

                print(output.shape)

            .. testoutput::

                (1, 10, 10, 64)

    .. [1] Image-to-Image Translation with Conditional Adversarial Networks
             https://arxiv.org/abs/1611.07004
    .. [2] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
             https://arxiv.org/abs/1711.11585
    """

    def __init__(
        self,
        eps: float = 1e-6,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
    ) -> None:
        r"""
        Initialize the layer.

        Args:
            eps (float): Variance_epsilon used by batch_norm layer.
            beta_initializer (str): Initializer for the beta variable.
            gamma_initializer (str): Initializer for the gamma variable.

        """
        super().__init__()
        self._eps = eps
        self.gamma = None
        self.beta = None
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Assemble the layer.

        Args:
            input_shape (:obj:`tuple` of (:obj:`int`)): Specifies the shape of the input
                accepted by the layer.

        """
        shape = (1, 1, 1, input_shape[-1])
        self.gamma = self.add_weight(
            shape=shape,
            initializer=self.gamma_initializer,
            name="gamma",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=shape, initializer=self.beta_initializer, name="beta", trainable=True
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Perform the computation.

        Args:
            inputs (:py:class:`tf.Tensor`): Inputs for the computation.
            training (bool): Controls for training or evaluation mode.

        Returns:
            :py:class:`tf.Tensor`: Output Tensor.

        """
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)

        return tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=self._eps,
        )
