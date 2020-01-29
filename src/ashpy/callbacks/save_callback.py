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

"""Save weights callback."""
import shutil
from collections import deque
from enum import Enum, Flag, auto
from pathlib import Path
from typing import List

import tensorflow as tf
from ashpy.callbacks import CounterCallback, Event


class SaveSubFormat(Enum):
    """Save Sub-Format enum."""

    TF = "tf"  #: TensorFlow format
    H5 = "h5"  #: H5 Format


class SaveFormat(Flag):
    """Save Format enum."""

    WEIGHTS = auto()  #: Weights format, saved using `model.save_weights()`
    MODEL = (
        auto()
    )  #: Model format (weights and architecture), saved using `model.save()`

    def name(self) -> str:
        """Name of the format."""
        if self == SaveFormat.WEIGHTS:
            return "weights"
        if self == SaveFormat.MODEL:
            return "saved-model"
        return "saved-model-and-weights"

    @staticmethod
    def _initialize_dirs(save_dir: Path, save_format, save_sub_format) -> Path:
        """Initialize the directory for this save_format and sub-format."""
        save_dir = save_dir / save_format.name()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        save_dir = (
            save_dir
            if save_sub_format == SaveSubFormat.TF
            else save_dir / save_format.name()
        )

        return save_dir

    def save(
        self,
        model: tf.keras.models.Model,
        save_dir: Path,
        save_sub_format: SaveSubFormat = SaveSubFormat.TF,
    ) -> None:
        """
        Save the model using the correct format and sub-format.

        Args:
            model (:py:class:`tf.keras.models.Model`): model to Save.
            save_dir (:class:`pathlib.Path`): path of the file in which to save the model.
            save_sub_format (:py:class:`ashpy.callbacks.save_callback.SaveSubFormat`): sub-format
                of the save operation.

        """
        if SaveFormat.WEIGHTS & self:

            save_dir = self._initialize_dirs(
                save_dir, SaveFormat.WEIGHTS, save_sub_format
            )
            # NOTE: Keras (TF 2.1.0) checks for h5 file using endswith attribute.
            # Explicit conversion to strings is required
            model.save_weights(str(save_dir), save_format=save_sub_format.value)

        if SaveFormat.MODEL & self:

            save_dir = self._initialize_dirs(
                save_dir, SaveFormat.MODEL, save_sub_format
            )
            # NOTE: TensorFlow 2.1.0 wanth either binary or unicod string.
            # Explicit conversion to strings is required
            model.save(str(save_dir), save_format=save_sub_format.value)

        if not (SaveFormat.MODEL & self) | (SaveFormat.WEIGHTS & self):
            raise NotImplementedError(
                "No implementation of `save` method for the current SaveFormat"
            )


class SaveCallback(CounterCallback):
    """
    Save Callback implementation.

    Examples:
        .. testcode::

            import shutil
            import operator
            import os

            generator = models.gans.ConvGenerator(
                layer_spec_input_res=(7, 7),
                layer_spec_target_res=(28, 28),
                kernel_size=(5, 5),
                initial_filters=32,
                filters_cap=16,
                channels=1,
            )

            discriminator = models.gans.ConvDiscriminator(
                layer_spec_input_res=(28, 28),
                layer_spec_target_res=(7, 7),
                kernel_size=(5, 5),
                initial_filters=16,
                filters_cap=32,
                output_shape=1,
            )

            models = [generator, discriminator]

            save_callback = callbacks.SaveCallback(save_dir="testlog/savedir",
                                                   models=models,
                                                   save_format=callbacks.SaveFormat.WEIGHTS,
                                                   save_sub_format=callbacks.SaveSubFormat.TF)

            # initialize trainer passing the save_callback

    """

    def __init__(
        self,
        save_dir: Path,
        models: List[tf.keras.models.Model],
        event: Event = Event.ON_EPOCH_END,
        event_freq: int = 1,
        max_to_keep: int = 1,
        save_format: SaveFormat = SaveFormat.WEIGHTS | SaveFormat.MODEL,
        save_sub_format: SaveSubFormat = SaveSubFormat.TF,
        verbose: int = 0,
        name: str = "SaveCallback",
    ):
        """
        Build a Save Callback.

        Save Callbacks are used to save the model on events.
        You can specify two different save formats: weights and model.
        At the same time you can specify two different save sub-formats: tf or h5.
        You will find the model saved in the save_dir under the directory weights or model.

        Args:
            save_dir (str): directory in which to save the weights or the model.
            models (List[:py:class:`tf.keras.models.Model`]): list of models to save.
            event (:py:class:`ashpy.callbacks.events.Event`): events on which to trigger the
                saving operation.
            event_freq (int): frequency of saving operation.
            name (str): name of the callback.
            verbose (int): verbosity of the callback (0 or 1).
            max_to_keep (int): maximum files to keep. If max_to_keep == 1 only the most recent
                file is kept. In general `max_to_keep` files are kept.
            save_format (:py:class:`ashpy.callbacks.save_callback.SaveFormat`): weights or model.
            save_sub_format (:py:class:`ashpy.callbacks.save_callback.SaveSubFormat`): sub-format
                of the saving (tf or h5).

        """
        super(SaveCallback, self).__init__(
            event, self.save_weights_fn, name, event_freq
        )
        self._save_dir = save_dir
        self._models = models
        self._verbose = verbose
        self._max_to_keep = max_to_keep

        if not isinstance(save_format, SaveFormat):
            raise TypeError("Use the SaveFormat enum!")

        self._save_format = save_format

        if not isinstance(save_sub_format, SaveSubFormat):
            raise TypeError("Use the SaveSubFormat enum!")

        self._save_sub_format = save_sub_format
        self._counter = 0
        self._save_path_histories = [deque() for _ in self._models]

        self._check_compatibility()

    def _check_compatibility(self):
        if (
            self._save_sub_format == SaveSubFormat.H5
            and self._save_format == SaveFormat.MODEL
        ):
            for model in self._models:
                if (
                    not model._is_graph_network  # pylint:disable=protected-access
                    and not isinstance(model, tf.keras.models.Sequential)
                ):
                    raise NotImplementedError(
                        "Saving the model to HDF5 format requires the model to be a "
                        "Functional model or a Sequential model. It does not work for "
                        "subclassed models, because such models are defined via the body of "
                        "a Python method, which isn't safely serializable. Consider saving "
                        "to the Tensorflow SavedModel format "
                        "(by setting save_sub_format=SaveSubFormat.TF) "
                        "or using save_format=SaveFormat.WEIGHTS."
                    )

    def _cleanup(self):
        """Cleanup stuff."""
        while self._counter > self._max_to_keep:
            for save_path_history in self._save_path_histories:
                if len(save_path_history) >= self._max_to_keep:
                    # Get the first element of the queue
                    save_dir_to_remove = save_path_history.popleft()

                    if self._verbose:
                        print(f"{self._name}: Removing {save_dir_to_remove} from disk.")

                    # Remove directory
                    shutil.rmtree(save_dir_to_remove, ignore_errors=True)

            # Decrease counter
            self._counter -= 1

    def _save_weights_fn(self, step: int):
        """
        Save weights.

        Args:
            step (int): current step.

        """
        for i, model in enumerate(self._models):

            if self._verbose:
                print(
                    f"{self._name}: {self._event.value} {self._event_counter.numpy()} - "
                    f"Saving model {i} to {self._save_dir} using format {self._save_format.name()} "
                    f"and sub-format {self._save_sub_format.value}."
                )

            # Create the correct directory name
            save_dir_i = self._save_dir / f"model-{i}-step-{step}"

            if not save_dir_i.exists():
                save_dir_i.mkdir(parents=True)

            # Add to the history
            self._save_path_histories[i].append(save_dir_i)

            # Save using the save_format
            self._save_format.save(
                model=model, save_dir=save_dir_i, save_sub_format=self._save_sub_format
            )

        # Increase the counter of saved files
        self._counter += 1

    def save_weights_fn(self, context):
        """Save weights and clean up if needed."""
        # Save weights phase
        self._save_weights_fn(context.global_step.numpy())

        # Clean up phase
        self._cleanup()
