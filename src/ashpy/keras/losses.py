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

"""Custom Keras losses, used by the AshPy executors."""

import tensorflow as tf


class L1(tf.keras.losses.Loss):
    """L1 Loss implementation as :py:class:`tf.keras.losses.Loss`."""

    def __init__(self) -> None:
        """Initialize the Loss."""
        super().__init__()
        self._reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    @property
    def reduction(self) -> tf.keras.losses.Reduction:
        """Return the current `reduction` for this type of loss."""
        return self._reduction

    @reduction.setter
    def reduction(self, value: tf.keras.losses.Reduction) -> None:
        """
        Set the `reduction`.

        Args:
            value (:py:class:`tf.keras.losses.Reduction`): Reduction to use for the loss.

        """
        self._reduction = value

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Compute the mean of the l1 between x and y."""
        if self._reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            axis = None
        elif self._reduction == tf.keras.losses.Reduction.NONE:
            axis = (1, 2, 3)
        else:
            raise ValueError("L1: unhandled reduction type ", self._reduction)

        return tf.reduce_mean(tf.abs(x - y), axis=axis)


class DMinMax(tf.keras.losses.Loss):
    r"""
    Implementation of MinMax Discriminator loss as :py:class:`tf.keras.losses.Loss`.

    .. math::
         L_{D} =  - \frac{1}{2} E [\log(D(x)) + \log (1 - D(G(z))]

    """

    def __init__(self, from_logits: bool = True, label_smoothing: float = 0.0) -> None:
        """Initialize the loss."""
        self._positive_bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            reduction=tf.keras.losses.Reduction.AUTO,
        )

        self._negative_bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=0.0,
            reduction=tf.keras.losses.Reduction.AUTO,
        )
        super().__init__()

    @property
    def reduction(self) -> tf.keras.losses.Reduction:
        """
        Return the reduction type of this loss.

        Returns:
            :py:classes:`tf.keras.losses.Reduction`: Reduction.

        """
        return self._positive_bce.reduction

    @reduction.setter
    def reduction(self, value: tf.keras.losses.Reduction) -> None:
        self._positive_bce.reduction = value
        self._negative_bce.reduction = value

    def call(self, d_real: tf.Tensor, d_fake: tf.Tensor) -> tf.Tensor:
        """
        Compute the MinMax Loss.

        Play the DiscriminatorMinMax game between the discriminator
        computed in real and the discriminator compute with fake inputs.

        Args:
            d_real (:py:class:`tf.Tensor`): Real data.
            d_fake (:py:class:`tf.Tensor`): Fake (generated) data.

        Returns:
            :py:class:`tf.Tensor`: Output Tensor.

        """
        return 0.5 * (
            self._positive_bce(tf.ones_like(d_real), d_real)
            + self._negative_bce(tf.zeros_like(d_fake), d_fake)
        )


class DLeastSquare(tf.keras.losses.Loss):
    """Discriminator Least Square Loss as :py:class:`tf.keras.losses.Loss`."""

    def __init__(self) -> None:
        r"""
        Least square Loss for Discriminator.

        Reference: Least Squares Generative Adversarial Networks [1]_ .

        Basically the Mean Squared Error between
        the discriminator output when evaluated in fake samples and 0
        and the discriminator output when evaluated in real samples and 1:
        For the unconditioned case this is:

        .. math::
            L_{D} = \frac{1}{2} E[(D(x) - 1)^2 + (0 - D(G(z))^2]

        where x are real samples and z is the latent vector.

        For the conditioned case this is:

        .. math::
            L_{D} = \frac{1}{2} E[(D(x, c) - 1)^2 + (0 - D(G(c), c)^2]

        where c is the condition and x are real samples.

        .. [1] Least Squares Generative Adversarial Networks https://arxiv.org/abs/1611.04076

        """
        self._positive_mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.AUTO
        )
        self._negative_mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.AUTO
        )
        super().__init__()

    @property
    def reduction(self) -> tf.keras.losses.Reduction:
        """
        Return the reduction type for this loss.

        Returns:
            :py:class:`tf.keras.losses.Reduction`: Reduction.

        """
        return self._positive_mse.reduction

    @reduction.setter
    def reduction(self, value) -> None:
        self._positive_mse.reduction = value
        self._negative_mse.reduction = value

    def call(self, d_real: tf.Tensor, d_fake: tf.Tensor) -> tf.Tensor:
        """
        Compute the Least Square Loss.

        Args:
            d_real (:py:class:`tf.Tensor`): Discriminator evaluated in real samples.
            d_fake (:py:class:`tf.Tensor`): Discriminator evaluated in fake samples.

        Returns:
            :py:class:`tf.Tensor`: Loss.

        """
        return 0.5 * (
            self._positive_mse(tf.ones_like(d_real), d_real)
            + self._negative_mse(tf.zeros_like(d_fake), d_fake)
        )


class DHingeLoss(tf.keras.losses.Loss):
    r"""
    Discriminator Hinge Loss as Keras Metric.

    See Geometric GAN [1]_ for more details.

    The Discriminator Hinge loss is the hinge version
    of the adversarial loss.
    The Hinge loss is defined as:

    .. math::
        L_{\text{hinge}} = \max(0, 1 -t y)

    where y is the Discriminator output
    and t is the target class (+1 or -1 in the case of binary classification).

    For the case of GANs:

    .. math::
        L_{D_{\text{hinge}}} = - \mathbb{E}_{(x,y) \sim p_data} [ \min(0, -1+D(x,y)) ] -
            \mathbb{E}_{x \sim p_x, y \sim p_data} [ \min(0, -1 - D(G(z),y)) ]

    .. [1] Geometric GAN https://arxiv.org/abs/1705.02894
    """

    def __init__(self) -> None:
        """Initialize the Loss."""
        self._hinge_loss_real = tf.keras.losses.Hinge(
            reduction=tf.keras.losses.Reduction.AUTO
        )
        self._hinge_loss_fake = tf.keras.losses.Hinge(
            reduction=tf.keras.losses.Reduction.AUTO
        )
        super().__init__()

    @property
    def reduction(self) -> tf.keras.losses.Reduction:
        """Return the current `reduction` for this type of loss."""
        return self._hinge_loss_fake.reduction

    @reduction.setter
    def reduction(self, value: tf.keras.losses.Reduction) -> None:
        """
        Set the `reduction`.

        Args:
            value (:py:class:`tf.keras.losses.Reduction`): Reduction to use for the loss.

        """
        self._hinge_loss_fake.reduction = value
        self._hinge_loss_real.reduction = value

    def call(self, d_real: tf.Tensor, d_fake: tf.Tensor) -> tf.Tensor:
        """Compute the hinge loss."""
        real_loss = self._hinge_loss_real(tf.ones_like(d_real), d_real)
        fake_loss = self._hinge_loss_fake(
            tf.math.negative(tf.ones_like(d_fake)), d_fake
        )

        loss = real_loss + fake_loss  # shape: (batch_size, 1)

        return loss


class GHingeLoss(tf.keras.losses.Loss):
    r"""
    Generator Hinge Loss as Keras Metric.

    See Geometric GAN [1]_ for more details.
    The Generator Hinge loss is the hinge version
    of the adversarial loss.
    The Hinge loss is defined as:

    .. math::
        L_{\text{hinge}} = \max(0, 1 - t y)

    where y is the Discriminator output
    and t is the target class (+1 or -1 in the case of binary classification).
    The target class of the generated images is +1.

    For the case of GANs

    .. math::
        L_{G_{\text{hinge}}} = - \mathbb{E}_{(x \sim p_x, y \sim p_data} [ \min(0, -1+D(G(x),y)) ]

    This can be simply approximated as:

    .. math::
        L_{G_{\text{hinge}}} = - \mathbb{E}_{(x \sim p_x, y \sim p_data} [ D(G(x),y) ]

    .. [1] Geometric GAN https://arxiv.org/abs/1705.02894

    """

    def __init__(self) -> None:
        """Initialize the Loss."""
        super().__init__()
        self._reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    @property
    def reduction(self) -> tf.keras.losses.Reduction:
        """Return the current `reduction` for this type of loss."""
        return self._reduction

    @reduction.setter
    def reduction(self, value: tf.keras.losses.Reduction) -> None:
        """
        Set the `reduction`.

        Args:
            value (:py:class:`tf.keras.losses.Reduction`): Reduction to use for the loss.

        """
        self._reduction = value

    def call(self, d_real: tf.Tensor, d_fake: tf.Tensor) -> tf.Tensor:
        """Compute the hinge loss."""
        return -d_fake
