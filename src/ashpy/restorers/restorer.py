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

"""Primitive Restorer, can be used standalone."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import ashpy
import tensorflow as tf

__ALL__ = ["Restorer"]


class Restorer:
    r"""
    :class:`Restorer` provide a way to restore objects from :class:`tf.train.Checkpoint`.

    Can be standalone.
    """

    def __init__(
        self,
        logdir: Union[Path, str] = Path().cwd() / "log",
        ckpts_dir: str = "ckpts",
        expect_partial: bool = True,
    ) -> None:
        """
        Initialize the Restorer.

        Args:
            logdir (str): Path to the directory with the logs.
            ckpts_dir (str): Name of the directory with the checkpoints to restore.
            expect_partial (bool): Whether to expect partial restoring or not. Default to true.
                For more information see the docs for :py:func:`tf.train.Checkpoint.restore()`.

        """
        self._ckpts_dir = Path(logdir) / ckpts_dir
        if not self._ckpts_dir.exists():
            raise FileNotFoundError(f"{ckpts_dir} does not exist.")
        self._restored_log_msg = "Restored {} from checkpoint {}."
        try:
            self._human_checkpoint_map: Optional[
                Dict[str, str]
            ] = self._read_human_checkpoint_map()
        except FileNotFoundError:
            self._human_checkpoint_map = None

    @property
    def checkpoint_map(self) -> Optional[Dict[str, str]]:
        """
        Get the map of the ids in the checkpoint.

        Map is a Dict where keys are the `ids` in the checkpoint and the values are the
            string representation of the types.

        Returns:
            Dict if the map is found, else None.

        """
        return self._human_checkpoint_map

    def _restore_checkpoint(self, checkpoint, partial: bool = True):
        """Restore or initialize the persistence layer (checkpoint)."""
        manager = tf.train.CheckpointManager(checkpoint, self._ckpts_dir, max_to_keep=3)
        if not manager.latest_checkpoint:
            raise FileNotFoundError(
                f"Could not find any checkpoint in {self._ckpts_dir}."
            )
        status = checkpoint.restore(manager.latest_checkpoint)
        if partial:
            status = status.expect_partial()
        status.assert_existing_objects_matched()
        return status

    @staticmethod
    def _validate_placeholder(placeholder: List, placeholder_type):
        # We do a preliminary check on types since the error thrown by TF can be hard to parse.
        try:
            assert isinstance(placeholder, placeholder_type)
        except AssertionError:
            raise TypeError(
                f"Object {placeholder} is should be of type: {placeholder_type}"
            )

    def restore_object(self, placeholder, object_ckpt_id: str):
        """
        Restore a placeholder from a checkpoint using the specified id.

        Warning:
            When restoring a :class:`tf.keras.Model` object from checkpoint assure that the
            model has been correctly built and instantiated by firstly calling it on some
            sample inputs. In the case of a model built with either the Sequential or
            Functional API an exception will be raised; for a model built with the Chainer API
            it will fail silently, restoration will be "successful" but no values will actually
            be restored since there are no valid placeholder as the model has not be built yet.

        TODO: Args
        TODO: Example
        """
        checkpoint = tf.train.Checkpoint(**{object_ckpt_id: placeholder})
        status = self._restore_checkpoint(checkpoint)
        print(self._restored_log_msg.format(object_ckpt_id, self._ckpts_dir))
        return status

    # The following methods are provided as convenience since these objects are stored in
    # the Checkpoint by the Trainer.
    def get_global_step(self) -> tf.Variable:
        """Return the restored global_step."""
        placeholder = tf.Variable(
            -1, name="global_step", trainable=False, dtype=tf.int64
        )
        assert self.restore_object(
            placeholder, ashpy.trainers.Trainer.ckpt_id_global_step
        )
        return placeholder

    def get_steps_per_epoch(self) -> tf.Variable:
        """Return the restored global_step."""
        placeholder = tf.Variable(
            -1, name="steps_per_epoch", trainable=False, dtype=tf.int64
        )
        assert self.restore_object(
            placeholder, ashpy.trainers.Trainer.ckpt_id_steps_per_epoch
        )
        return placeholder

    def restore_callback(
        self, callback: ashpy.callbacks.Callback, callback_ckpt_id: str
    ) -> List[ashpy.callbacks.Callback]:
        """Return the restored callbacks."""
        self._validate_placeholder(callback, ashpy.callbacks.Callback)
        assert self.restore_object(callback, callback_ckpt_id)
        return callback

    def _read_human_checkpoint_map(self) -> Dict[str, str]:
        with open(self._ckpts_dir / "checkpoint_map.json") as fp:
            return json.load(fp)
