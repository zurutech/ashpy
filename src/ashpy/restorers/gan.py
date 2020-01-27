# Copyright 2020 Zuru Tech HK Limited. All Rights Reserved.
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

"""Convenience :class:`Restorer` to be used with :mod:`ashpy.trainers.gan` ."""

import tensorflow as tf
from ashpy.restorers.restorer import Restorer
from ashpy.trainers import AdversarialTrainer, EncoderTrainer

__ALL__ = ["AdversarialRestorer", "AdversarialEncoderRestorer"]


class AdversarialRestorer(Restorer):
    """Convenience :class:`Restorer` for ease of use with the :class:`AdversairalTrainer`."""

    def restore_generator(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Restore the Generator model.

        Args:
            model (:class:`tf.keras.Model`): The placeholder model in which values from the
                checkpoint will be restored.

        Returns:
            Restored model.

        Warning:
            When restoring a :class:`tf.keras.Model` object from checkpoint assure that the
            model has been correctly built and instantiated by firstly calling it on some
            sample inputs. In the case of a model built with either the Sequential or
            Functional API an exception will be raised; for a model built with the Chainer API
            it will fail silently, restoration will be "successful" but no values will actually
            be restored since there are no valid placeholder as the model has not be built yet.

        """
        self.restore_object(model, AdversarialTrainer.ckpt_id_generator)
        return model

    def restore_discriminator(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Restore the Discriminator model.

        Args:
            model (:class:`tf.keras.Model`): The placeholder model in which values from the
                checkpoint will be restored.

        Returns:
            Restored model.

        Warning:
            When restoring a :class:`tf.keras.Model` object from checkpoint assure that the
            model has been correctly built and instantiated by firstly calling it on some
            sample inputs. In the case of a model built with either the Sequential or
            Functional API an exception will be raised; for a model built with the Chainer API
            it will fail silently, restoration will be "successful" but no values will actually
            be restored since there are no valid placeholder as the model has not be built yet.

        """
        self.restore_object(model, AdversarialTrainer.ckpt_id_discriminator)
        return model

    def restore_generator_optimizer(
        self, optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.keras.optimizers.Optimizer:
        """
        Restore the Optimizer used to train the Generator model.

        Args:
            model (:class:`tf.keras.optimizers.Optimizer`): The placeholder Optimizer in
                which values from the checkpoint will be restored.

        Returns:
            Restored optimizer.

        """
        self.restore_object(optimizer, AdversarialTrainer.ckpt_id_optimizer_generator)
        return optimizer

    def restore_discriminator_optimizer(
        self, optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.keras.optimizers.Optimizer:
        """
        Restore the Optimizer used to train the Discriminator model.

        Args:
            model (:class:`tf.keras.optimizers.Optimizer`): The placeholder Optimizer in
                which values from the checkpoint will be restored.

        Returns:
            Restored optimizer.

        """
        self.restore_object(
            optimizer, AdversarialTrainer.ckpt_id_optimizer_discriminator
        )
        return optimizer


class AdversarialEncoderRestorer(AdversarialRestorer):
    """Convenience :class:`Restorer` for ease of use with the :class:`EncoderTrainer`."""

    def restore_encoder(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Restore the Encoder model.

        Args:
            model (:class:`tf.keras.Model`): The placeholder model in which values from the
                checkpoint will be restored.

        Returns:
            Restored model.

        Warning:
            When restoring a :class:`tf.keras.Model` object from checkpoint assure that the
            model has been correctly built and instantiated by firstly calling it on some
            sample inputs. In the case of a model built with either the Sequential or
            Functional API an exception will be raised; for a model built with the Chainer API
            it will fail silently, restoration will be "successful" but no values will actually
            be restored since there are no valid placeholder as the model has not be built yet.

        """
        self.restore_object(model, EncoderTrainer.ckpt_id_encoder)
        return model

    def restore_encoder_optimizer(
        self, optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.keras.optimizers.Optimizer:
        """
        Restore the Optimizer used to train the Encoder model.

        Args:
            model (:class:`tf.keras.optimizers.Optimizer`): The placeholder Optimizer in
                which values from the checkpoint will be restored.

        Returns:
            Restored optimizer.

        """
        self.restore_object(optimizer, EncoderTrainer.ckpt_id_optimizer_encoder)
        return optimizer
