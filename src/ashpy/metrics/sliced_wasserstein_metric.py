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

"""Sliced Wasserstein Distance metric."""
from __future__ import annotations

import operator
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import tensorflow as tf
from ashpy import LogEvalMode
from ashpy.metrics import Metric
from ashpy.metrics.sliced_wasserstein import sliced_wasserstein_distance

if TYPE_CHECKING:
    from ashpy.contexts import GANContext  # pylint: disable=ungrouped-imports

__ALL__ = ["SingleSWD", "SlicedWassersteinDistance"]


class SingleSWD(Metric):
    """SlicedWassersteinDistance for a certain level of the pyramid."""

    def __init__(
        self,
        model_selection_operator: Callable = operator.lt,
        logdir: Union[Path, str] = Path().cwd() / "log",
        level_of_pyramid: int = 0,
        real_or_fake: str = "fake",
    ) -> None:
        """
        Initialize the Metric.

        Args:
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an operator is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.

            level_of_pyramid (int): Level of the pyramid related to this metric.
            real_or_fake (str): string identifying this metric (real or fake distance).

        """
        super().__init__(
            name=f"SWD_{level_of_pyramid}_{real_or_fake}",
            metric=tf.metrics.Mean(
                name=f"SWD_{level_of_pyramid}_{real_or_fake}", dtype=tf.float32
            ),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

    def update_state(self, context: GANContext, score: Union[float, tf.Tensor]) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANContext`): An AshPy Context
                Object that carries all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        self._distribute_strategy.experimental_run_v2(updater(score))


class SlicedWassersteinDistance(Metric):
    r"""
    Sliced Wasserstein Distance.

    Used as metric in Progressive Growing of GANs [1]_.

    .. [1] Progressive Growing of GANs https://arxiv.org/abs/1710.10196

    """

    def __init__(
        self,
        name: str = "SWD",
        model_selection_operator: Callable = operator.lt,
        logdir: Union[Path, str] = Path().cwd() / "log",
        resolution: int = 128,
        resolution_min: int = 16,
        patches_per_image: int = 64,
        patch_size: int = 7,
        random_sampling_count: int = 1,
        random_projection_dim: int = 7 * 7 * 3,
        use_svd: bool = False,
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
            resolution (int): Image Resolution, defaults to 128
            resolution_min (int): Min Resolution achieved by the metric
            patches_per_image (int): Number of patches to extract per image per Laplacian level.
            patch_size (int): Width of a square patch.
            random_sampling_count (int): Number of random projections to average.
            random_projection_dim (int): Dimension of the random projection space.
            use_svd (bool): experimental method to compute a more accurate distance.

        """
        super().__init__(
            name=name,
            metric=tf.metrics.Mean(name=name, dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        if resolution <= resolution_min:
            raise ValueError("Minimum resolution cannot be smaller than the resolution")

        self.resolution = resolution
        self.resolution_min = resolution_min
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.random_sampling_count = random_sampling_count
        self.random_projection_dim = random_projection_dim
        self.use_svd = use_svd

        self.children_real_fake = [
            (
                SingleSWD(model_selection_operator, logdir, 2 ** i, "real"),
                SingleSWD(model_selection_operator, logdir, 2 ** i, "fake"),
            )
            for i in range(
                int(np.log2(resolution)), int(np.log2(resolution_min)) - 1, -1
            )
        ]

    @property
    def logdir(self) -> str:
        """Retrieve the log directory."""
        return self._logdir

    @logdir.setter
    def logdir(self, logdir) -> None:
        """Set the logdir changing also other properties."""
        self._logdir = logdir
        self._update_logdir()
        for child_metric_real, child_metric_fake in self.children_real_fake:
            child_metric_real.logdir, child_metric_fake.logdir = logdir, logdir

    def update_state(self, context: GANContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.gan.GANContext`): An AshPy Context Object
                that carries all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy

            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            # check the resolution is the same as the one passed as input
            resolution = real_x.shape[1]
            if resolution != self.resolution:
                raise ValueError(
                    "Image resolution is not the same as the input resolution."
                )

            scores = sliced_wasserstein_distance(
                real_x,
                fake,
                resolution_min=self.resolution_min,
                patches_per_image=self.patches_per_image,
                use_svd=self.use_svd,
                patch_size=self.patch_size,
                random_projection_dim=self.random_projection_dim,
                random_sampling_count=self.random_sampling_count,
            )

            fake_scores = []

            for i, couple in enumerate(scores):
                self.children_real_fake[i][0].update_state(context, couple[0])
                self.children_real_fake[i][1].update_state(context, couple[1])
                fake_scores.append(tf.expand_dims(couple[1], axis=0))

            fake_scores = tf.concat(fake_scores, axis=0)

            self._distribute_strategy.experimental_run_v2(updater(fake_scores))

    def model_selection(
        self, checkpoint: tf.train.Checkpoint, global_step: tf.Variable
    ) -> None:
        """Perform model selection for each sub-metric."""
        super().model_selection(checkpoint, global_step)
        for child in self.children_real_fake:
            child[0].model_selection(checkpoint, global_step)
            child[1].model_selection(checkpoint, global_step)

    def log(self, step):
        """Log the SWD mean and each sub-metric."""
        # log mean
        tf.summary.scalar(self.name, self.result(), step=step)
        # call log method of each child
        for child in self.children_real_fake:
            child[0].log(step)
            child[1].log(step)

    def reset_states(self) -> None:
        """Reset the state of the metric and the state of each child metric."""
        self._metric.reset_states()
        # for each child log
        for child in self.children_real_fake:
            child[0].reset_states()
            child[1].reset_states()
