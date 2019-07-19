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

"""Attention Layer implementation."""

import tensorflow as tf


class Attention(tf.keras.Model):
    r"""
    Attention Layer from Self-Attention GAN [1]_.

    First we extract features from the previous layer:

    .. math::
        f(x) = W_f x

    .. math::
        g(x) = W_g x

    .. math::
        h(x) = W_h x

    Then we calculate the importance matrix:

    .. math::
        \beta_{j,i} = \frac{\exp(s_{i,j})}{\sum_{i=1}^{N}\exp(s_{ij})}

    :math:`\beta_{j,i}` indicates the extent to which the model attends to the :math:`i^{th}`
    location when synthethizing the :math:`j^{th}` region.

    Then we calculate the output of the attention layer
    :math:`(o_1, ..., o_N) \in \mathbb{R}^{C \times N}`:

    .. math::
        o_j = \sum_{i=1}^{N} \beta_{j,i} h(x_i)

    Finally we combine the (scaled) attention and the input to get the final output of
    the layer:

    .. math::
        y_i = \gamma o_i + x_i

    where :math:`\gamma` is initialized as 0.

    Examples:
        * Direct Usage:

            .. testcode::

                x = tf.ones((1, 10, 10, 64))

                # instantiate attention layer as model
                attention = Attention(64)

                # evaluate passing x
                output = attention(x)

                # the output shape is
                # the same as the input shape
                print(output.shape)

        * Inside a Model:

            .. testcode::

                def MyModel():
                    inputs = tf.keras.layers.Input(shape=[None, None, 64])
                    attention = Attention(64)
                    return tf.keras.Model(inputs=inputs, outputs=attention(inputs))

                x = tf.ones((1, 10, 10, 64))
                model = MyModel()
                output = model(x)

                print(output.shape)

            .. testoutput::

                (1, 10, 10, 64)

    .. [1] Self-Attention Generative Adversarial Networks https://arxiv.org/abs/1805.08318

    """

    def __init__(self, filters: int) -> None:
        """
        Build the Attention Layer.

        Args:
            filters (int): Number of filters of the input tensor.
                It should be preferably a multiple of 8.

        """
        super().__init__()
        initializer = tf.random_normal_initializer(0.0, 0.02)

        self.f_conv = tf.keras.layers.Conv2D(
            filters // 8, 1, strides=1, padding="same", kernel_initializer=initializer
        )
        self.g_conv = tf.keras.layers.Conv2D(
            filters // 8, 1, strides=1, padding="same", kernel_initializer=initializer
        )
        self.h_conv = tf.keras.layers.Conv2D(
            filters, 1, strides=1, padding="same", kernel_initializer=initializer
        )

        self.gamma = tf.Variable(0, dtype=tf.float32)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Perform the computation.

        Args:
            inputs (:py:class:`tf.Tensor`): Inputs for the computation.
            training (bool): Controls for training or evaluation mode.

        Returns:
            :py:class:`tf.Tensor`: Output Tensor.

        """
        f = self.f_conv(inputs)
        g = self.g_conv(inputs)
        h = self.h_conv(inputs)

        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(beta, h)
        x = self.gamma * o + inputs
        return x
