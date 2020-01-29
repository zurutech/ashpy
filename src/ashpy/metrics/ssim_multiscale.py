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

"""Multiscale Structural Similarity metric."""
from __future__ import annotations

import math
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Tuple, Union

import tensorflow as tf
from ashpy import LogEvalMode
from ashpy.metrics import Metric

if TYPE_CHECKING:
    from ashpy.contexts import GANContext  # pylint: disable=ungrouped-imports

__ALL__ = ["SSIM_Multiscale"]


class SSIM_Multiscale(Metric):  # pylint: disable=invalid-name
    r"""
    Multiscale Structural Similarity.

    See Multiscale structural similarity for image quality assessment [1]_

    .. [1] Multiscale structural similarity for image quality assessment
        https://ieeexplore.ieee.org/document/1292216

    """

    def __init__(
        self,
        name: str = "SSIM_Multiscale",
        model_selection_operator: Callable = operator.lt,
        logdir: Union[Path, str] = Path().cwd() / "log",
        max_val: float = 2.0,
        power_factors=None,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        """
        Initialize the Metric.

        Args:
            name (str): Name of the metric.
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an operator is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.
            max_val (float): The dynamic range of the images
                (i.e., the difference between the maximum the and minimum)
                (see www.tensorflow.org/versions/r2.0/api_docs/python/tf/image/ssim_multiscale)
            power_factors (List[float]): Iterable of weights for each of the scales. The number of
                scales used is the length of the list. Index 0 is the unscaled
                resolution's weight and each increasing scale corresponds to the image
                being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
                0.1333), which are the values obtained in the original paper.
            filter_size (int): Default value 11 (size of gaussian filter).
            filter_sigma (float): Default value 1.5 (width of gaussian filter).
            k1 (float): Default value 0.01.
            k2 (float): Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
                it would be better if we take the values in range of 0< K2 <0.4).

        """
        super().__init__(
            name=name,
            metric=tf.metrics.Mean(name=name, dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1  # pylint: disable=invalid-name
        self.k2 = k2  # pylint: disable=invalid-name
        if power_factors is None:
            power_factors = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.power_factors = power_factors

        self.max_val = max_val

    def update_state(self, context: GANContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANContext`): An AshPy Context Object
                that carries all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        for real_xy, noise in context.dataset:
            _, real_y = real_xy

            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            img1, img2 = self.split_batch(fake)

            ssim_multiscale = tf.image.ssim_multiscale(
                img1,
                img2,
                max_val=self.max_val,
                power_factors=self.power_factors,
                filter_sigma=self.filter_sigma,
                filter_size=self.filter_size,
                k1=self.k1,
                k2=self.k2,
            )

            self._distribute_strategy.experimental_run_v2(updater(ssim_multiscale))

    @staticmethod
    def split_batch(batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split a batch along axis 0 into two tensors having the same size.

        Args:
            batch (tf.Tensor): A batch of images.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor]) The batch split in two tensors.

        Raises:
            ValueError: if the batch has size 1.

        """
        batch_size = batch.shape[0]
        if tf.equal(tf.math.mod(batch_size, 2), 0):
            return batch[: batch_size // 2, :, :, :], batch[batch_size // 2 :, :, :, :]
        split_value = math.floor(batch_size / 2)
        if split_value == 0:
            raise ValueError(
                "Batch size too small."
                "You can use SSIM_MULTISCALE metric only with batch size > 1"
            )
        return batch[:split_value, :, :, :], batch[split_value:, :, :, :]
