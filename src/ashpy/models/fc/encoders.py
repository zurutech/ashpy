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

"""Collection of Encoders (i.e., GANs' Discriminators) models."""
from ashpy.models.fc.interfaces import FCInterface
from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["Encoder"]


class Encoder(FCInterface):
    """
    Primitive Model for all fully connected encoder based architecture.

    Examples:
        .. testcode::

            encoder = Encoder(
                hidden_units=[256,128,64],
                output_shape=10)
            print(encoder(tf.zeros((1,55))).shape)

        .. testoutput::

            (1, 10)

    """

    def __init__(self, hidden_units, output_shape):
        """
        Instantiate the :py:class:`Decoder`.

        Args:
            hidden_units (:obj:`tuple` of :obj:`int`): Number of units per hidden layer.
            output_shape (int): Amount of units of the last :py:obj:`tf.keras.layers.Dense`.

        Returns:
            :py:obj:`None`

        """
        super().__init__()

        # Assembling Model
        for units in hidden_units:
            self.model_layers.extend(
                [
                    keras.layers.Dense(units),
                    keras.layers.LeakyReLU(),
                    keras.layers.Dropout(0.3),
                ]
            )
        self.model_layers.append(keras.layers.Dense(output_shape))
