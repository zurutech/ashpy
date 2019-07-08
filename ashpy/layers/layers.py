#Copyright 2019 Zuru Tech HK Limited. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""
Ash Custom layers.

.. currentmodule:: ashpy.layers

.. rubric:: Layers

.. autosummary::
    :nosignatures:
    :toctree: layers

    InstanceNormalization
    Attention

"""
import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    r"""
    Instance Normalization Layer (used by Pix2Pix [1]_ and Pix2PixHD [2]_ )
    Basically it's a batch normalization done at instance level.
    The implementation follows the basic implementation of the Batch Normalization Layer.

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
    ):
        r"""

        Args:
            eps (float): variance_epsilon used by batch_norm layer
            beta_initializer (str): initializer for the beta variable
            gamma_initializer (str): initializer for the gamma variable
        """
        super().__init__()
        self._eps = eps
        self.gamma = None
        self.beta = None
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

    def build(self, input_shape):
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

    def call(self, inputs, training=False):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)

        return tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=self._eps,
        )


class Attention(tf.keras.Model):
    r"""
    Attention Layer from Self-Attention GAN [1]_

    .. [1] Self-Attention Generative Adversarial Networks https://arxiv.org/abs/1805.08318

    """

    def __init__(self, filters: int):
        """
        Builds the Attention Layer

        Args:
            filters (int): number of filters of the input tensor
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

    def call(self, inputs, training=False):
        f = self.f_conv(inputs)
        g = self.g_conv(inputs)
        h = self.h_conv(inputs)

        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(beta, h)
        x = self.gamma * o + inputs
        return x
