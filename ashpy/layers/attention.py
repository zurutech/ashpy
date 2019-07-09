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

import tensorflow as tf


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
