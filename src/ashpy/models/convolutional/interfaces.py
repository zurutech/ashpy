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

"""Primitive Convolutional interfaces."""

import inspect

import numpy as np
from tensorflow import keras  # pylint: disable=no-name-in-module

__ALL__ = ["Conv2DInterface"]


class Conv2DInterface(keras.Model):
    """Primitive Interface to be used by all :mod:`ashpy.models`."""

    def __init__(self):
        """
        Primitive Interface to be used by all :mod:`ashpy.models`.

        Declares the `self.model_layers` list.
        """
        super().__init__()
        self.model_layers = []

    @staticmethod
    def _get_layer_spec(initial_filers, filters_cap, input_res, target_res):
        """
        Compose the ``layer_spec``, the building block of a convolutional model.

        The ``layer_spec`` is an iterator. Every element returned is the number of filters
        to learn for the current layer.
        The generated sequence of filters starts ``from initial_filters`` and halve/double
        the number of filters depending on the ``input_res`` and ``target_res``.
        If ``input_res > target_res`` the number of filters increases, else it decreases.
        The progression is always a power of 2.

        Args:
            initial_filers (int): Depth of the first convolutional layer.
            filters_cap (int): Maximum number of filters per layer.
            input_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Input resolution.
            target_res (:obj:`tuple` of (:obj:`int`, :obj:`int`)): Output resolution.

        Yields:
            int: Number of filters to use for the conv layer.

        Examples:
            .. testcode::

                # Encoder
                class T(Conv2DInterface):
                    pass
                spec = T._get_layer_spec(
                    initial_filers=16,
                    filters_cap=128,
                    input_res=(512, 256),
                    target_res=(32, 16)
                )
                print([s for s in spec])

                spec = T._get_layer_spec(
                    initial_filers=16,
                    filters_cap=128,
                    input_res=(28, 28),
                    target_res=(7, 7)
                )
                print([s for s in spec])

                # Decoder
                spec = T._get_layer_spec(
                    initial_filers=128,
                    filters_cap=16,
                    input_res=(32, 16),
                    target_res=(512, 256)
                )
                print([s for s in spec])

                spec = T._get_layer_spec(
                    initial_filers=128,
                    filters_cap=16,
                    input_res=(7, 7),
                    target_res=(28, 28)
                )
                print([s for s in spec])

            .. testoutput::

                [32, 64, 128, 128]
                [32, 64]
                [64, 32, 16, 16]
                [64, 32]


        Notes:
            This is useful since it enables us to dynamically redefine models sharing
            an underlying architecture but with different resolutions.

        """
        if isinstance(input_res, int):
            input_res = (input_res, input_res)

        if isinstance(target_res, int):
            target_res = (target_res, target_res)

        end_res, start_res = min(*target_res), min(*input_res)
        num_layers = int(np.floor(np.log2(end_res)) - np.floor(np.log2(start_res)))
        s = np.sign(num_layers)
        op = np.max if s > 0 else np.min
        start = 0
        stop = int(abs(num_layers))
        for layer_id in range(start, stop):
            filters = int(np.floor(initial_filers * (2.0 ** (-s * (layer_id + 1)))))
            filters = op((filters_cap, filters))
            yield filters

    def call(self, inputs, training=True, return_features=False):
        """
        Execute the model on input data.

        Args:
            inputs (:py:class:`tf.Tensor`): Input tensor(s).
            training (:obj:`bool`): Training flag.
            return_features (:obj:`bool`): If True returns the features.

        Returns:
            :py:class:`tf.Tensor`: The model output.

        """
        layer_input = inputs
        features = []
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

            if isinstance(layer, keras.layers.LeakyReLU):
                features.append(layer_input)

        if return_features:
            return layer_input, features
        return layer_input
