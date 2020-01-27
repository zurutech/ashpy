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

"""Test Restorers."""

from pathlib import Path

import pytest
import tensorflow as tf

from ashpy.restorers import (
    AdversarialEncoderRestorer,
    AdversarialRestorer,
    ClassifierRestorer,
    Restorer,
)
from ashpy.trainers import AdversarialTrainer, ClassifierTrainer, Trainer
from tests.utils.fake_datasets import (
    fake_adversarial_dataset,
    fake_autoencoder_datasest,
)
from tests.utils.fake_models import basic_dcgan, conv_autoencoder
from tests.utils.fake_training_loop import fake_classifier_training_loop

DEFAULT_CKPT_DIR = "ckpts"


class TestRestorer:
    """
    Test the Restorer.

    GIVEN a correctly instantiated trainer
    GIVEN some training has been done
        WHEN calling the Restorer "Restored ... from checkpoint: .../ckpts/ckpt-..."
            should be logged.
        WHEN restoring models the first layer of the restored model and the trained one should
            have the same weights.

    """

    def _check_first_layer(
        self, trained: tf.keras.Model, restored: tf.keras.Model, i=0
    ):
        """Test that the first layers of the restored and trained model have the same weights."""
        try:
            try:
                trained_layer = trained.layers[i].weights[0]
                restored_layers = restored.layers[i].weights[0]
            except IndexError:
                trained_layer = trained.layers[i].weights
                restored_layers = restored.layers[i].weightss
            assert tf.reduce_all(tf.equal(trained_layer, restored_layers))
        except AttributeError:
            i += 1
            print(f"Proceed to layer in position {i}")
            self._check_first_layer(trained, restored, i)

    def _test_restore_object(self, restorer, placeholder, ckpt_id, capsys):
        """Test that the object is restored correctly."""
        restorer.restore_object(placeholder, ckpt_id)
        self._check_log(restorer, ckpt_id, capsys)

    def _check_log(self, restorer, ckpt_id, capsys):
        """Test that the object is restored correctly by looking at the logs."""
        out, err = capsys.readouterr()
        # Assert that the log is correct
        assert restorer._restored_log_msg.format(
            ckpt_id, restorer._ckpts_dir
        ) in out.split("\n")

    def test_restore_model(self, fake_training, capsys, tmpdir):
        """
        Test that models are correctly restored.

        The test is performed by checking the logs and the first layer of each model.
        """
        training_loop, loop_args, metrics = fake_training
        training_completed, trainer = training_loop(
            logdir=tmpdir, metrics=metrics, **loop_args
        )
        assert training_completed
        restorer = Restorer(logdir=tmpdir)

        if isinstance(trainer, ClassifierTrainer):

            placeholder = conv_autoencoder()

            # Ensure model have been built correctly
            x, y = next(iter(fake_autoencoder_datasest()))
            placeholder(x)

            ckpt_id = trainer.ckpt_id_model
            self._test_restore_object(restorer, placeholder, ckpt_id, capsys)
            self._check_first_layer(trainer._model, placeholder)

        elif isinstance(trainer, AdversarialTrainer):

            placeholder_g, placeholder_d = basic_dcgan(**loop_args)

            # Ensure model have been built correctly
            (x, y), z = next(iter(fake_adversarial_dataset(**loop_args)))
            fake = placeholder_g(z)
            assert tf.reduce_all(tf.equal(fake.shape, x.shape))
            placeholder_d(x)

            self._test_restore_object(
                restorer, placeholder_g, trainer.ckpt_id_generator, capsys
            )
            self._check_first_layer(trainer._generator, placeholder_g)
            self._test_restore_object(
                restorer, placeholder_d, trainer.ckpt_id_discriminator, capsys
            )
            self._check_first_layer(trainer._discriminator, placeholder_d)

    def test_restore_common_variables(self, fake_training, capsys, tmpdir):
        """
        Test that the convenience methods exposed by :class:`Restorer` work correctly.

        The common :class:`tf.Variable`s that can be restored from :class:`Restorer` are:
            - global step
            - steps per epoch

        """
        training_loop, loop_args, metrics = fake_training
        training_completed, trainer = training_loop(
            logdir=tmpdir, metrics=metrics, **loop_args
        )
        assert training_completed
        restorer = Restorer(logdir=tmpdir)

        # Restore variables and check their values using the convenience method
        assert tf.equal(trainer._global_step, restorer.get_global_step())
        assert tf.equal(trainer._steps_per_epoch, restorer.get_steps_per_epoch())

        out, err = capsys.readouterr()

        # Check the log
        for id_to_check in [
            trainer.ckpt_id_global_step,
            trainer.ckpt_id_steps_per_epoch,
        ]:
            # Assert that the log is correct
            assert restorer._restored_log_msg.format(
                id_to_check, restorer._ckpts_dir
            ) in out.split("\n")

    # ###################################################3
    # Test Convenience Methods

    def _test_convenience_model_restorer(
        self,
        restorer: AdversarialRestorer,
        convenience_method,
        placeholder_model,
        trained_model,
        ckpt_id,
        capsys,
    ):
        convenience_method(placeholder_model)
        self._check_log(restorer, ckpt_id, capsys)
        self._check_first_layer(trained_model, placeholder_model)

    def _test_convenience_optimizer_restorer(
        self, restorer, convenience_method, placeholder_optimizer, ckpt_id, capsys
    ):
        """
        Test that the various optimizers are correctly restored using convenience classes.

        TODO: Add a more thorough check like :meth:`_check_first_layer()`
        """
        convenience_method(placeholder_optimizer)
        self._check_log(restorer, ckpt_id, capsys)

    def test_convenience_restorer(self, fake_training, capsys, tmpdir):
        """
        Test that models and optimizers are correctly restored using the convenience classes.

        TODO: Add test for AdversarialEncoderRestorer
        """
        logdir = tmpdir
        training_loop, loop_args, metrics = fake_training
        training_completed, trainer = training_loop(
            logdir=logdir, metrics=metrics, **loop_args
        )
        assert training_completed

        if isinstance(trainer, ClassifierTrainer):
            restorer: ClassifierRestorer = ClassifierRestorer(logdir=logdir)

            placeholder_model = conv_autoencoder()
            placeholder_opt = tf.keras.optimizers.Adam()

            # Ensure model have been built correctly
            x, y = next(iter(fake_autoencoder_datasest()))
            placeholder_model(x)

            ckpt_id = trainer.ckpt_id_model
            self._test_convenience_model_restorer(
                restorer,
                restorer.restore_model,
                placeholder_model,
                trainer._model,
                trainer.ckpt_id_model,
                capsys,
            )
            self._test_convenience_optimizer_restorer(
                restorer,
                restorer.restore_optimizer,
                placeholder_opt,
                trainer.ckpt_id_optimizer,
                capsys,
            )

        elif isinstance(trainer, AdversarialTrainer):
            restorer: AdversarialRestorer = AdversarialRestorer(logdir=logdir)

            placeholder_g, placeholder_d = basic_dcgan(**loop_args)
            placeholder_optimizer_g, placeholder_optimizer_d = (
                tf.keras.optimizers.Adam(),
                tf.keras.optimizers.Adam(),
            )

            # Ensure model have been built correctly
            (x, y), z = next(iter(fake_adversarial_dataset(**loop_args)))
            fake = placeholder_g(z)
            assert tf.reduce_all(tf.equal(fake.shape, x.shape))
            placeholder_d(x)

            self._test_convenience_model_restorer(
                restorer,
                restorer.restore_generator,
                placeholder_g,
                trainer._generator,
                trainer.ckpt_id_generator,
                capsys,
            )
            self._test_convenience_optimizer_restorer(
                restorer,
                restorer.restore_generator_optimizer,
                placeholder_optimizer_g,
                trainer.ckpt_id_optimizer_generator,
                capsys,
            )
            self._test_convenience_model_restorer(
                restorer,
                restorer.restore_discriminator,
                placeholder_d,
                trainer._discriminator,
                trainer.ckpt_id_discriminator,
                capsys,
            )
            self._test_convenience_optimizer_restorer(
                restorer,
                restorer.restore_discriminator_optimizer,
                placeholder_optimizer_g,
                trainer.ckpt_id_optimizer_discriminator,
                capsys,
            )
