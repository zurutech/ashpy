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

"""GAN losses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Type, Union

import tensorflow as tf
from ashpy.contexts import GANContext, GANEncoderContext
from ashpy.keras.losses import L1, DHingeLoss, DLeastSquare, DMinMax, GHingeLoss
from ashpy.losses.executor import Executor, SumExecutor

if TYPE_CHECKING:
    from ashpy.ashtypes import TWeight


class AdversarialLossType(Enum):
    """Enumeration for Adversarial Losses. Implemented: GAN and LSGAN."""

    GAN = auto()  # classical gan loss (minmax)
    LSGAN = auto()  # Least Square GAN
    HINGE_LOSS = auto()  # Hinge loss


class GANExecutor(Executor, ABC):
    """
    Executor for GANs.

    Implements the basic functions needed by the GAN losses.
    """

    @abstractmethod
    def call(self, context, **kwargs):
        """
        Execute the function, using the information provided by the context.

        Args:
            context (:py:class:`ashpy.contexts.Context`): The function
                execution Context.

        Returns:
            :py:obj:`tf.Tensor`: Output Tensor.

        """
        super(GANExecutor, self).call(context, **kwargs)

    @staticmethod
    def get_discriminator_inputs(
        context: GANContext,
        fake_or_real: tf.Tensor,
        condition: tf.Tensor,
        training: bool,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """
        Return the discriminator inputs. If needed it uses the encoder.

        The current implementation uses the number of inputs to determine
        whether the discriminator is conditioned or not.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANContext`): Context for GAN models.
            fake_or_real (:py:class:`tf.Tensor`): Discriminator input tensor,
                it can be fake (generated) or real.
            condition (:py:class:`tf.Tensor`): Discriminator condition
                (it can also be generator noise).
            training (:py:class:`bool`): whether is training phase or not

        Returns:
            The discriminator inputs.

        """
        num_inputs = len(context.discriminator_model.inputs)

        # Handle Encoder
        if isinstance(context, GANEncoderContext):
            if num_inputs == 2:
                d_inputs = [
                    fake_or_real,
                    context.encoder_model(fake_or_real, training=training),
                ]
            elif num_inputs == 3:
                d_inputs = [
                    fake_or_real,
                    context.encoder_model(fake_or_real, training=training),
                    condition,
                ]
            else:
                raise ValueError(
                    f"Context has encoder_model, but generator has only {num_inputs} inputs"
                )
        else:
            if num_inputs == 2:
                d_inputs = [fake_or_real, condition]
            else:
                d_inputs = fake_or_real

        return d_inputs


class GeneratorAdversarialLoss(GANExecutor):
    r"""Base class for the adversarial loss of the generator."""

    def __init__(self, loss_fn: tf.keras.losses.Loss = None) -> None:
        """
        Initialize the Executor.

        Args:
            loss_fn (:py:class:`tf.keras.losses.Loss`): Keras Loss function to call
                passing (tf.ones_like(d_fake_i), d_fake_i).

        """
        super().__init__(loss_fn)

    @Executor.reduce_loss
    def call(
        self,
        context: GANContext,
        *,
        fake: tf.Tensor,
        condition: tf.Tensor,
        training: bool,
        **kwargs,
    ) -> tf.Tensor:
        r"""
        Configure the discriminator inputs and calls `loss_fn`.

        Args:
            context (:py:class:`ashpy.contexts.GANContext`): GAN Context.
            fake (:py:class:`tf.Tensor`): Fake images.
            condition (:py:class:`tf.Tensor`): Generator conditioning.
            training (bool): If training or evaluation.

        Returns:
            :py:class:`tf.Tensor`: The loss for each example.

        """
        fake_inputs = self.get_discriminator_inputs(
            context=context, fake_or_real=fake, condition=condition, training=training
        )

        d_fake = context.discriminator_model(fake_inputs, training=training)

        # Support for Multiscale discriminator
        # TODO: Improve
        if isinstance(d_fake, list):
            value = tf.add_n(
                [
                    tf.reduce_mean(
                        self._fn(tf.ones_like(d_fake_i), d_fake_i), axis=[1, 2]
                    )
                    for d_fake_i in d_fake
                ]
            )
            return value

        value = self._fn(tf.ones_like(d_fake), d_fake)
        value = tf.cond(
            tf.equal(tf.rank(d_fake), tf.constant(4)),
            lambda: value,
            lambda: tf.expand_dims(tf.expand_dims(value, axis=-1), axis=-1),
        )
        return tf.reduce_mean(value, axis=[1, 2])


class GeneratorBCE(GeneratorAdversarialLoss):
    r"""
    The Binary CrossEntropy computed among the generator and the 1 label.

    .. math::
        L_{G} =  E [\log (D( G(z))]

    """

    def __init__(self, from_logits: bool = True) -> None:
        """Initialize the BCE Loss for the Generator."""
        self.name = "GeneratorBCE"
        super().__init__(tf.keras.losses.BinaryCrossentropy(from_logits=from_logits))


class GeneratorLSGAN(GeneratorAdversarialLoss):
    r"""
    Least Square GAN Loss for generator.

    Reference: https://arxiv.org/abs/1611.04076

    .. note::
        Basically the Mean Squared Error between the discriminator output when evaluated
        in fake and 1.

    .. math::
        L_{G} =  \frac{1}{2} E [(1 - D(G(z))^2]

    """

    def __init__(self) -> None:
        """Initialize the Least Square Loss for the Generator."""
        super().__init__(tf.keras.losses.MeanSquaredError())
        self.name = "GeneratorLSGAN"


class GeneratorL1(GANExecutor):
    r"""
    L1 loss between the generator output and the target.

    .. math::
        L_G = E ||x - G(z)||_1

    Where x is the target and G(z) is generated image.

    """

    def __init__(self) -> None:
        """Initialize the Executor."""
        super().__init__(L1())

    @Executor.reduce_loss
    def call(self, context: GANContext, *, fake: tf.Tensor, real: tf.Tensor, **kwargs):
        """
        Call the carried loss on `fake` and `real`.

        Args:
            context (:py:class:`ashpy.contexts.GANContext`): GAN Context.
            fake (:py:class:`tf.Tensor`): Fake data (generated).
            real (:py:class:`tf.Tensor`): Real data.

        Returns:
            :py:class:`tf.Tensor`: Output Tensor.

        """
        mae = self._fn(fake, real)
        return mae


class GeneratorHingeLoss(GeneratorAdversarialLoss):
    r"""
    Hinge loss for the Generator.

    See Geometric GAN [1]_ for more details.

    .. [1] Geometric GAN https://arxiv.org/abs/1705.02894
    """

    def __init__(self) -> None:
        """Initialize the Least Square Loss for the Generator."""
        super().__init__(GHingeLoss())
        self.name = "GeneratorHingeLoss"


class FeatureMatchingLoss(GANExecutor):
    r"""
    Conditional GAN Feature matching loss.

    The loss is computed for each example and it's the L1 (MAE) of the feature difference.
    Implementation of pix2pix HD: https://github.com/NVIDIA/pix2pixHD

    .. math::
        \text{FM} = \sum_{i=0}^N \frac{1}{M_i} ||D_i(x, c) - D_i(G(c), c) ||_1

    Where:

    - D_i is the i-th layer of the discriminator
    - N is the total number of layer of the discriminator
    - M_i is the number of components for the i-th layer
    - x is the target image
    - c is the condition
    - G(c) is the generated image from the condition c
    - || ||_1 stands for norm 1.

    This is for a single example: basically for each layer
    of the discriminator we compute the absolute error between
    the layer evaluated in real examples and in fake examples.
    Then we average along the batch. In the case where D_i is
    a multidimensional tensor we simply calculate the mean
    over the axis 1,2,3.
    """

    def __init__(self) -> None:
        """Initialize the Executor."""
        super().__init__(L1())

    @Executor.reduce_loss
    def call(
        self,
        context: GANContext,
        *,
        fake: tf.Tensor,
        real: tf.Tensor,
        condition: tf.Tensor,
        training: bool,
        **kwargs,
    ) -> tf.Tensor:
        """
        Configure the discriminator inputs and calls `loss_fn`.

        Args:
            context (:py:class:`ashpy.contexts.GANContext`): GAN Context.
            fake (:py:class:`tf.Tensor`): Fake data.
            real (:py:class:`tf.Tensor`): Real data.
            condition (:py:class:`tf.Tensor`): Generator conditioning.
            training (bool): If training or evaluation.

        Returns:
            :py:class:`tf.Tensor`: The loss for each example.

        """
        fake_inputs = self.get_discriminator_inputs(
            context, fake_or_real=fake, condition=condition, training=training
        )

        real_inputs = self.get_discriminator_inputs(
            context, fake_or_real=real, condition=condition, training=training
        )

        _, features_fake = context.discriminator_model(
            fake_inputs, training=training, return_features=True
        )
        _, features_real = context.discriminator_model(
            real_inputs, training=training, return_features=True
        )

        # for each feature the L1 between the real and the fake
        # every call to fn should return [batch_size, 1] that is the mean L1
        feature_loss = [
            self._fn(feat_real_i, feat_fake_i)
            for feat_real_i, feat_fake_i in zip(features_real, features_fake)
        ]
        mae = tf.add_n(feature_loss)
        return mae


class CategoricalCrossEntropy(Executor):
    r"""
    Categorical Cross Entropy between generator output and target.

    Useful when the output of the generator is a distribution over classes.

    ..note::
        The target must be represented in one hot notation.

    """

    def __init__(self) -> None:
        """Initialize the Categorical Cross Entropy Executor."""
        self.name = "CrossEntropy"
        super().__init__(tf.keras.losses.CategoricalCrossentropy())

    @Executor.reduce_loss
    def call(self, context: GANContext, *, fake: tf.Tensor, real: tf.Tensor, **kwargs):
        """
        Compute the categorical cross entropy loss.

        Args:
            context: Unused.
            fake (:py:class:`tf.Tensor`): Fake data G(condition).
            real (:py:class:`tf.Tensor`): Real data x(c).

        Returns:
            The categorical cross entropy loss for each example.

        """
        loss_value = tf.reduce_mean(self._fn(real, fake), axis=[1, 2])
        return loss_value


class Pix2PixLoss(SumExecutor):
    r"""
    Pix2Pix Loss.

    Weighted sum of :py:class:`ashpy.losses.gan.GeneratorL1`,
    :py:class:`ashpy.losses.gan.AdversarialLossG` and
    :py:class:`ashpy.losses.gan.FeatureMatchingLoss`.

    Used by Pix2Pix [1] and Pix2PixHD [2]

    .. [1] Image-to-Image Translation with Conditional Adversarial Networks
             https://arxiv.org/abs/1611.07004
    .. [2] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
             https://arxiv.org/abs/1711.11585

    """

    def __init__(
        self,
        l1_loss_weight: TWeight = 100.0,
        adversarial_loss_weight: TWeight = 1.0,
        feature_matching_weight: TWeight = 10.0,
        adversarial_loss_type: Union[
            AdversarialLossType, int
        ] = AdversarialLossType.GAN,
        use_feature_matching_loss: bool = False,
    ) -> None:
        r"""
        Initialize the loss.

        Weighted sum of :py:class:`ashpy.losses.gan.GeneratorL1`,
        :py:class:`ashpy.losses.gan.AdversarialLossG` and
        :py:class:`ashpy.losses.gan.FeatureMatchingLoss`.

        Args:
            l1_loss_weight (:py:obj:`ashpy.ashtypes.TWeight`): Weight of L1 loss.
            adversarial_loss_weight (:py:obj:`ashpy.ashtypes.TWeight`): Weight of adversarial loss.
            feature_matching_weight (:py:obj:`ashpy.ashtypes.TWeight`): Weight of the
                feature matching loss.
            adversarial_loss_type (:py:class:`ashpy.losses.gan.AdversarialLossType`): Adversarial
                loss type (:py:class:`ashpy.losses.gan.AdversarialLossType.GAN`
                or :py:class:`ashpy.losses.gan.AdversarialLossType.LSGAN`).
            use_feature_matching_loss (bool): if True use also uses
                :py:class:`ashpy.losses.gan.FeatureMatchingLoss`.

        """
        executors = [
            GeneratorL1() * l1_loss_weight,
            get_adversarial_loss_generator(adversarial_loss_type)()
            * adversarial_loss_weight,
        ]

        if use_feature_matching_loss:
            executors.append(FeatureMatchingLoss() * feature_matching_weight)

        super().__init__(executors)


class Pix2PixLossSemantic(SumExecutor):
    r"""
    Semantic Pix2Pix Loss.

    Weighted sum of :py:class:`ashpy.losses.gan.CategoricalCrossEntropy`,
    :py:class:`ashpy.losses.gan.AdversarialLossG` and
    :py:class:`ashpy.losses.gan.FeatureMatchingLoss`.

    """

    def __init__(
        self,
        cross_entropy_weight: TWeight = 100.0,
        adversarial_loss_weight: TWeight = 1.0,
        feature_matching_weight: TWeight = 10.0,
        adversarial_loss_type: AdversarialLossType = AdversarialLossType.GAN,
        use_feature_matching_loss: bool = False,
    ):
        r"""
        Initialize the Executor.

        Weighted sum of :py:class:`ashpy.losses.gan.CategoricalCrossEntropy`,
        :py:class:`ashpy.losses.gan.AdversarialLossG`
        and :py:class:`ashpy.losses.gan.FeatureMatchingLoss`

        Args:
            cross_entropy_weight (:py:obj:`ashpy.ashtypes.TWeight`): Weight of the categorical
                cross entropy loss.
            adversarial_loss_weight (:py:obj:`ashpy.ashtypes.TWeight`): Weight of the
                adversarial loss.
            feature_matching_weight (:py:obj:`ashpy.ashtypes.TWeight`): Weight of the
                feature matching loss.
            adversarial_loss_type (:py:class:`ashpy.losses.gan.AdversarialLossType`): type of
                adversarial loss, see :py:class:`ashpy.losses.gan.AdversarialLossType`
            use_feature_matching_loss (bool): whether to use feature matching loss or not

        """
        executors = [
            CategoricalCrossEntropy() * cross_entropy_weight,
            get_adversarial_loss_generator(adversarial_loss_type)()
            * adversarial_loss_weight,
        ]

        if use_feature_matching_loss:
            executors.append(FeatureMatchingLoss() * feature_matching_weight)
        super().__init__(executors)


# TODO: Check if this supports condition
class EncoderBCE(Executor):
    """The Binary Cross Entropy computed among the encoder and the 0 label."""

    def __init__(self, from_logits: bool = True) -> None:
        """Initialize the Executor."""
        super().__init__(tf.keras.losses.BinaryCrossentropy(from_logits=from_logits))

    @Executor.reduce_loss
    def call(
        self, context: GANEncoderContext, *, real: tf.Tensor, training: bool, **kwargs
    ):
        """
        Compute the Encoder BCE.

        Args:
            context (:py:class:`ashpy.contexts.GANEncoderContext`): GAN Context
                with Encoder support.
            real (:py:class:`tf.Tensor`): Real images.
            training (bool): If training or evaluation.

        Returns:
            :py:class:`tf.Tensor`: The loss for each example.

        """
        encode = context.encoder_model(real, training=training)
        d_real = context.discriminator_model([real, encode], training=training)
        return self._fn(tf.zeros_like(d_real), d_real)


class DiscriminatorAdversarialLoss(GANExecutor):
    r"""Base class for the adversarial loss of the discriminator."""

    def __init__(self, loss_fn: tf.keras.losses.Loss = None) -> None:
        r"""
        Initialize the Executor.

        Args:
            loss_fn (:py:class:`tf.keras.losses.Loss`): Loss function call passing
            (d_real, d_fake).

        """
        super().__init__(loss_fn)

    @Executor.reduce_loss
    def call(
        self,
        context: GANContext,
        *,
        fake: tf.Tensor,
        real: tf.Tensor,
        condition: tf.Tensor,
        training: bool,
        **kwargs,
    ):
        r"""
        Call: setup the discriminator inputs and calls `loss_fn`.

        Args:
            context (:py:class:`ashpy.contexts.GANContext`): GAN Context.
            fake (:py:class:`tf.Tensor`): Fake images corresponding to the condition G(c).
            real (:py:class:`tf.Tensor`): Real images corresponding to the condition x(c).
            condition (:py:class:`tf.Tensor`): Condition for the generator and discriminator.
            training (bool): if training or evaluation

        Returns:
            :py:class:`tf.Tensor`: The loss for each example.

        """
        fake_inputs = self.get_discriminator_inputs(
            context, fake_or_real=fake, condition=condition, training=training
        )

        real_inputs = self.get_discriminator_inputs(
            context, fake_or_real=real, condition=condition, training=training
        )

        d_fake = context.discriminator_model(fake_inputs, training=training)
        d_real = context.discriminator_model(real_inputs, training=training)

        if isinstance(d_fake, list):
            value = tf.add_n(
                [
                    tf.reduce_mean(self._fn(d_real_i, d_fake_i), axis=[1, 2])
                    for d_real_i, d_fake_i in zip(d_real, d_fake)
                ]
            )
            return value
        value = self._fn(d_real, d_fake)
        value = tf.cond(
            tf.equal(tf.rank(d_fake), tf.constant(4)),
            lambda: value,
            lambda: tf.expand_dims(tf.expand_dims(value, axis=-1), axis=-1),
        )
        return tf.reduce_mean(value, axis=[1, 2])


class DiscriminatorMinMax(DiscriminatorAdversarialLoss):
    r"""
    The min-max game played by the discriminator.

    .. math::
        L_{D} =  - \frac{1}{2} E [\log(D(x)) + \log (1 - D(G(z))]

    """

    def __init__(self, from_logits=True, label_smoothing=0.0):
        """Initialize Loss."""
        super().__init__(
            DMinMax(from_logits=from_logits, label_smoothing=label_smoothing)
        )


class DiscriminatorLSGAN(DiscriminatorAdversarialLoss):
    r"""
    Least square Loss for discriminator.

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

    .. [1] Least Squares Generative Adversarial Networks
        https://arxiv.org/abs/1611.04076

    """

    def __init__(self) -> None:
        """Initialize loss."""
        super().__init__(DLeastSquare())
        self.name = "DiscriminatorLSGAN"


class DiscriminatorHingeLoss(DiscriminatorAdversarialLoss):
    r"""
    Hinge loss for the Discriminator.

    See Geometric GAN [1]_ for more details.

    .. [1] Geometric GAN https://arxiv.org/abs/1705.02894
    """

    def __init__(self) -> None:
        """Initialize the Least Square Loss for the Generator."""
        super().__init__(DHingeLoss())
        self.name = "DiscriminatorHingeLoss"


###
# Utility functions in order to get the correct loss
###


def get_adversarial_loss_discriminator(
    adversarial_loss_type: Union[AdversarialLossType, int] = AdversarialLossType.GAN
) -> Type[Executor]:
    r"""
    Return the correct loss fot the Discriminator.

    Args:
        adversarial_loss_type (:py:class:`ashpy.losses.gan.AdversarialLossType`): Type of loss
            (:py:class:`ashpy.losses.gan.AdversarialLossType.GAN` or
            :py:class:`ashpy.losses.gan.AdversarialLossType.LSGAN`)

    Returns:
        The correct (:py:class:`ashpy.losses.executor.Executor`) (to be instantiated).

    """
    if adversarial_loss_type == AdversarialLossType.GAN:
        return DiscriminatorMinMax
    if adversarial_loss_type == AdversarialLossType.LSGAN:
        return DiscriminatorLSGAN
    if adversarial_loss_type == AdversarialLossType.HINGE_LOSS:
        return DiscriminatorHingeLoss
    raise ValueError(
        "Loss type not supported, the implemented losses are GAN, LSGAN or HINGE_LOSS."
    )


def get_adversarial_loss_generator(
    adversarial_loss_type: Union[AdversarialLossType, int] = AdversarialLossType.GAN
) -> Type[Executor]:
    r"""
    Return the correct loss for the Generator.

    Args:
        adversarial_loss_type (:py:class:`ashpy.losses.AdversarialLossType`): Type of loss
            (:py:class:`ashpy.losses.AdversarialLossType.GAN` or
            :py:class:`ashpy.losses.AdversarialLossType.LSGAN`).

    Returns:
        The correct (:py:class:`ashpy.losses.executor.Executor`) (to be instantiated).

    """
    if adversarial_loss_type == AdversarialLossType.GAN:
        return GeneratorBCE
    if adversarial_loss_type == AdversarialLossType.LSGAN:
        return GeneratorLSGAN
    if adversarial_loss_type == AdversarialLossType.HINGE_LOSS:
        return GeneratorHingeLoss
    raise ValueError(
        "Loss type not supported, the implemented losses are GAN, LSGAN or HINGE_LOSS."
    )
