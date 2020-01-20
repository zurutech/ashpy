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

"""Primitive Fully Connected interfaces."""

import inspect

from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["FCInterface"]


class FCInterface(keras.Model):
    """Primitive Interface to be used by all :mod:`ashpy.models`."""

    def __init__(self):
        """
        Primitive Interface to be used by all :mod:`ashpy.models`.

        Declares the self._layers list.

        Returns:
            :py:obj:`None`

        """
        super().__init__()
        self.model_layers = []

    def call(self, inputs, training=True):
        """
        Execute the model on input data.

        Args:
            inputs (:py:class:`tf.Tensor`): Input tensors.
            training (:obj:`bool`): Training flag.

        Returns:
            :py_class:`tf.Tensor`

        """
        layer_input = inputs
        for layer in self.model_layers:
            spec = inspect.getfullargspec(layer.call)

            dp_len = len(spec.defaults) if spec.defaults else 0
            args_count = len(spec.args)
            args = {}
            for i in range(args_count - dp_len, args_count):
                args[str(spec.args[i])] = spec.defaults[i - args_count]

            if "self" in args:
                del args["self"]

            if "training" in args:
                args["training"] = training
            layer_input = layer(layer_input, **args)
        return layer_input
