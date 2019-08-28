from __future__ import annotations

import operator
import os
from typing import TYPE_CHECKING, Union

from collections import Callable

import numpy as np
import tensorflow as tf

from ashpy.metrics import Metric
from ashpy.metrics.sliced_wassersein import sliced_wasserstein_distance
from ashpy import LogEvalMode


if TYPE_CHECKING:
    from ashpy.contexts import GANContext  # pylint: disable=ungrouped-imports


class SingleSWD(Metric):
    def __init__(
        self,
        model_selection_operator: Callable = operator.lt,
        logdir: str = os.path.join(os.getcwd(), "log"),
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

            level_of_pyramid (int): Level of the pyramid related to this metric

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
            context (:py:class:`ashpy.contexts.GANContext`): An AshPy Context Object that carries
                all the information the Metric needs.

        """
        updater = lambda value: lambda: self._metric.update_state(value)
        self._distribute_strategy.experimental_run_v2(updater(score))


class SlicedWasserseinDistance(Metric):
    """
    Sliced Wasserstein Distance.
    Used as metric in Progressive Growing of GANs (https://arxiv.org/abs/1710.10196)
    """

    def __init__(
        self,
        model_selection_operator: Callable = operator.lt,
        logdir: str = os.path.join(os.getcwd(), "log"),
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
            model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                be used when `model_selection` is triggered to compare the metrics,
                used by the `update_state`.
                Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.

                .. note::
                    Model selection is done ONLY if an operator is specified here.

            logdir (str): Path to the log dir, defaults to a `log` folder in the current
                directory.
            patches_per_image: (int) Number of patches to extract per image per Laplacian level.
            patch_size: (int) Width of a square patch.
            random_sampling_count: (int) Number of random projections to average.
            random_projection_dim: (int) Dimension of the random projection space.
            use_svd (bool): experimental method to compute a more accurate distance.

        """
        super().__init__(
            name="SWD",
            metric=tf.metrics.Mean(name="SWD", dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

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

    def update_state(self, context: GANContext) -> None:
        """
        Update the internal state of the metric, using the information from the context object.

        Args:
            context (:py:class:`ashpy.contexts.GANContext`): An AshPy Context Object that carries
                all the information the Metric needs.

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

    def model_selection(self, checkpoint: tf.train.Checkpoint) -> None:
        super().model_selection(checkpoint)
        for child in self.children_real_fake:
            child[0].model_selection(checkpoint)
            child[1].model_selection(checkpoint)

    def log(self, step):
        # log mean
        tf.summary.scalar(self.name, self.result(), step=step)
        # for each child log
        for child in self.children_real_fake:
            child[0].log(step)
            child[1].log(step)

    def reset_states(self) -> None:
        """Reset the state of the metric."""
        self._metric.reset_states()
        # for each child log
        for child in self.children_real_fake:
            child[0].reset_states()
            child[1].reset_states()
