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
import os
import shutil
from collections import deque
from enum import Enum, Flag, auto
from typing import List

import tensorflow as tf

from ashpy.callbacks import CounterCallback, Event


class SaveSubFormat(Enum):
    """Save Sub-Format enum."""

    TF = "tf"
    H5 = "h5"


class SaveFormat(Flag):
    """Save Format enum."""

    WEIGHTS = auto()
    MODEL = auto()

    def name(self) -> str:
        """Name of the format."""
        if self == SaveFormat.WEIGHTS:
            return "weights"
        if self == SaveFormat.MODEL:
            return "saved-model"
        return "saved-model-and-weights"

    @staticmethod
    def _initialize_dirs(save_dir, save_format, save_sub_format):
        """Initialize the directory for this save_format and sub-format."""
        save_dir = os.path.join(save_dir, save_format.name())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_dir = (
            save_dir
            if save_sub_format == SaveSubFormat.TF
            else os.path.join(save_dir, save_format.name())
        )
        return save_dir

    def save(
        self,
        model: tf.keras.models.Model,
        save_dir: str,
        save_sub_format: SaveSubFormat = SaveSubFormat.TF,
    ) -> None:
        """
        Save the model using the correct format and sub-format.

        Args:
            model (:py:class:`tf.keras.models.Model`): model to Save.
            save_dir (str): path of the file in which to save the model.
            save_sub_format (:py:class:`ashpy.callbacks.save_callback.SaveSubFormat`): sub-format
                of the save operation.

        """
        if SaveFormat.WEIGHTS & self:

            save_dir = self._initialize_dirs(
                save_dir, SaveFormat.WEIGHTS, save_sub_format
            )
            model.save_weights(save_dir, save_format=save_sub_format.value)

        if SaveFormat.MODEL & self:

            save_dir = self._initialize_dirs(
                save_dir, SaveFormat.MODEL, save_sub_format
            )
            model.save(save_dir, save_format=save_sub_format.value)

        if not ((SaveFormat.MODEL & self) | (SaveFormat.WEIGHTS & self)):
            raise NotImplementedError(
                "No implementation of `save` method for the current SaveFormat"
            )


class SaveCallback(CounterCallback):
    """Save Callback implementation."""

    def __init__(
        self,
        save_dir: str,
        models: List[tf.keras.models.Model],
        event: Event = Event.ON_EPOCH_END,
        event_freq: int = 1,
        name: str = "SaveCallback",
        verbose: int = 0,
        max_to_keep: int = 1,
        save_format: SaveFormat = SaveFormat.WEIGHTS | SaveFormat.MODEL,
        save_sub_format: SaveSubFormat = SaveSubFormat.TF,
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
            event (:py:class:`ashpy.callbacks.events.Event`): events on which to trigger the saving operation.
            event_freq (int): frequency of saving operation.
            name (str): name of the callback.
            verbose (int): verbosity of the callback (0 or 1).
            max_to_keep (int): maximum files to keep.
            save_format (:py:class:`ashpy.callbacks.save_callback.SaveFormat`): weights or model.
            save_sub_format (:py:class:`ashpy.callbacks.save_callback.SaveSubFormat`): sub-format of
                the saving (tf or h5).

        """
        super(SaveCallback, self).__init__(
            event, self.save_weights_fn, name, event_freq
        )
        self._save_dir = save_dir
        self._models = models
        self._verbose = verbose
        self._max_to_keep = max_to_keep
        self._save_format = save_format
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
                        "to the Tensorflow SavedModel format (by setting save_sub_format=SaveSubFormat.TF) "
                        "or using save_format=SaveFormat.WEIGHTS."
                    )

    def _cleanup(self):
        """Cleanup stuff."""
        while self._counter > self._max_to_keep:
            for save_path_history in self._save_path_histories:
                if len(save_path_history) >= self._max_to_keep:
                    # get the first element of the queue
                    save_dir_to_remove = save_path_history.popleft()

                    if self._verbose:
                        print(f"{self._name}: Removing {save_dir_to_remove} from disk.")

                    # remove directory
                    shutil.rmtree(save_dir_to_remove, ignore_errors=True)

            # decrease counter
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

            # create the correct directory name
            save_dir_i = os.path.join(self._save_dir, f"model-{i}-step-{step}")

            if not os.path.exists(save_dir_i):
                os.makedirs(save_dir_i)

            # add to the history
            self._save_path_histories[i].append(save_dir_i)

            # save using the save_format
            self._save_format.save(
                model=model, save_dir=save_dir_i, save_sub_format=self._save_sub_format
            )

        # increase the counter of saved files
        self._counter += 1

    def save_weights_fn(self, context):
        """Save weights and clean up if needed."""
        # save weights phase
        self._save_weights_fn(context.global_step.numpy())

        # clean up phase
        self._cleanup()
