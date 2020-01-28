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

"""Convenience :class:`Restorer` to be used with :mod:`ashpy.trainers.classifier` ."""

import tensorflow as tf
from ashpy.restorers.restorer import Restorer
from ashpy.trainers import ClassifierTrainer

__ALL__ = ["ClassifierRestorer"]


class ClassifierRestorer(Restorer):
    """Convenience :class:`Restorer` for ease of use with the :class:`ClassifierTrainer`."""

    def restore_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Restore the Classifier model.

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
        self.restore_object(model, ClassifierTrainer.ckpt_id_model)
        return model

    def restore_optimizer(
        self, optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.keras.optimizers.Optimizer:
        """
        Restore the Optimizer used to train the Classifier model.

        Args:
            model (:class:`tf.keras.optimizers.Optimizer`): The placeholder Optimizer in
                which values from the checkpoint will be restored.

        Returns:
            Restored optimizer.

        """
        self.restore_object(optimizer, ClassifierTrainer.ckpt_id_optimizer)
        return optimizer
