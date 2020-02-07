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

"""
Test Restorers.

GIVEN a correctly instantiated trainer
GIVEN some training has been done
    WHEN calling the Restorer "Restored ... from checkpoint: .../ckpts/ckpt-..."
        should be logged.
    WHEN restoring models the first layer of the restored model and the trained one should
        have the same weights.
"""
from pathlib import Path
from typing import Union

import pytest
import tensorflow as tf
from ashpy.callbacks import CounterCallback
from ashpy.restorers import (
    AdversarialRestorer,
    ClassifierRestorer,
    ModelNotConstructedError,
    Restorer,
)
from ashpy.trainers import AdversarialTrainer, ClassifierTrainer

from tests.utils.fake_training_loop import (
    FakeAdversarialTraining,
    FakeClassifierTraining,
)

DEFAULT_CKPT_DIR = "ckpts"


def _check_models_weights(trained: tf.keras.Model, restored: tf.keras.Model, i=0):
    """Test that the first layers of the restored and trained model have the same weights."""

    try:
        for i, element in enumerate(trained.weights):
            assert tf.reduce_all(tf.equal(element, restored.weights[i]))
    except AssertionError:
        raise ModelNotConstructedError


def _test_restore_object(restorer, placeholder, ckpt_id, capsys):
    """Test that the object is restored correctly."""
    restorer.restore_object(placeholder, ckpt_id)
    _check_log(restorer, ckpt_id, capsys)


def _check_log(restorer, ckpt_id, capsys):
    """Test that the object is restored correctly by looking at the logs."""
    out, _ = capsys.readouterr()
    # Assert that the log is correct
    assert restorer._restored_log_msg.format(ckpt_id, restorer._ckpts_dir) in out.split(
        "\n"
    )


def test_restore_model(fake_training_fn, capsys, tmpdir):
    """
    Test that models are correctly restored.

    The test is performed by checking the logs and the first layer of each model.
    """
    logdir = Path(tmpdir).joinpath("training")
    _tmp_logdir = Path(tmpdir).joinpath("banana")

    fake_training = fake_training_fn(logdir=logdir)
    assert fake_training()
    trainer = fake_training.trainer
    restorer = Restorer(logdir=logdir)

    if isinstance(trainer, ClassifierTrainer):

        new_loop = fake_training_fn(logdir=_tmp_logdir)
        placeholder = new_loop.model

        # Ensure model have been built correctly
        x, _ = next(iter(new_loop.dataset))
        placeholder(x)

        ckpt_id = trainer.ckpt_id_model
        _test_restore_object(restorer, placeholder, ckpt_id, capsys)
        _check_models_weights(trainer._model, placeholder)

    elif isinstance(trainer, AdversarialTrainer):

        new_loop: FakeAdversarialTraining = fake_training_fn(logdir=_tmp_logdir)
        placeholder_g, placeholder_d = (new_loop.generator, new_loop.discriminator)

        # Ensure that the ModelNotConstructedError is correctly triggered
        with pytest.raises(ModelNotConstructedError):
            _test_restore_object(
                restorer, placeholder_g, trainer.ckpt_id_generator, capsys
            )
            _test_restore_object(
                restorer, placeholder_d, trainer.ckpt_id_discriminator, capsys
            )

        # Ensure model have been built correctly
        (x, _), z = next(iter(new_loop.dataset))
        fake = placeholder_g(z)
        assert tf.reduce_all(tf.equal(fake.shape, x.shape))
        placeholder_d(x)

        _test_restore_object(restorer, placeholder_g, trainer.ckpt_id_generator, capsys)
        _check_models_weights(trainer._generator, placeholder_g)
        _test_restore_object(
            restorer, placeholder_d, trainer.ckpt_id_discriminator, capsys
        )
        _check_models_weights(trainer._discriminator, placeholder_d)


def test_restore_common_variables(fake_training_fn, capsys, tmpdir):
    """
    Test that the convenience methods exposed by :class:`Restorer` work correctly.

    The common :class:`tf.Variable`s that can be restored from :class:`Restorer` are:
        - global step
        - steps per epoch

    """
    logdir = Path(tmpdir).joinpath("training")

    fake_training = fake_training_fn(logdir=logdir)
    assert fake_training()
    trainer = fake_training.trainer
    restorer = Restorer(logdir=logdir)

    # Restore variables and check their values using the convenience method
    assert tf.equal(trainer._global_step, restorer.get_global_step())
    assert tf.equal(trainer._steps_per_epoch, restorer.get_steps_per_epoch())

    out, _ = capsys.readouterr()

    # Check the log
    for id_to_check in [
        trainer.ckpt_id_global_step,
        trainer.ckpt_id_steps_per_epoch,
    ]:
        # Assert that the log is correct
        assert restorer._restored_log_msg.format(
            id_to_check, restorer._ckpts_dir
        ) in out.split("\n")


def test_restore_callbacks(fake_training_fn, capsys, tmpdir):
    """Test that callbacks are succesfully restored."""
    logdir = Path(tmpdir).joinpath("training")

    fake_training = fake_training_fn(logdir=logdir)
    assert fake_training()
    trainer = fake_training.trainer
    restorer = Restorer(logdir=logdir)

    if isinstance(trainer, AdversarialTrainer):
        placeholder_callbacks = fake_training.callbacks
        for i, placeholder_callback in enumerate(placeholder_callbacks):
            # Restore the callbacks
            restorer.restore_callback(placeholder_callback, placeholder_callback.name)
            # Check the log
            _check_log(restorer, placeholder_callback.name, capsys)
            # Check that the trained values and the restored are equal
            if isinstance(placeholder_callback, CounterCallback):
                assert tf.equal(
                    placeholder_callbacks[i]._event_counter,
                    trainer._callbacks[i]._event_counter,
                )


def test_read_checkpoint_map(fake_training_fn, tmpdir):
    """Test that checkpoint map is read correctly."""
    logdir = Path(tmpdir).joinpath("training")

    fake_training = fake_training_fn(logdir=logdir)
    assert fake_training()
    trainer = fake_training.trainer
    restorer = Restorer(logdir=logdir)
    assert restorer.checkpoint_map == trainer._generate_checkpoint_map()

    # Test that Restorer.checkpoint_map without the checkpoint_map.json correctly returns None
    # Remove checkpoint_map.json
    ckpt_map: Path = restorer._ckpts_dir / "checkpoint_map.json"
    ckpt_map.unlink()
    assert not ckpt_map.exists()
    assert not Restorer(logdir).checkpoint_map


# ###################################################
# Test Convenience Methods


def _test_convenience_model_restorer(
    restorer: AdversarialRestorer,
    convenience_method,
    placeholder_model,
    trained_model,
    ckpt_id,
    capsys,
):
    convenience_method(placeholder_model)
    _check_log(restorer, ckpt_id, capsys)
    _check_models_weights(trained_model, placeholder_model)


def _test_convenience_optimizer_restorer(
    restorer, convenience_method, placeholder_optimizer, ckpt_id, capsys
):
    """
    Test that the various optimizers are correctly restored using convenience classes.

    TODO: Add a more thorough check like :meth:`_check_first_layer()`
    """
    convenience_method(placeholder_optimizer)
    _check_log(restorer, ckpt_id, capsys)


def test_convenience_restorer(fake_training_fn, capsys, tmpdir):
    """
    Test that models and optimizers are correctly restored using the convenience classes.

    TODO: Add test for AdversarialEncoderRestorer
    """
    logdir = Path(tmpdir).joinpath("training")
    _tmp_logdir = Path(tmpdir).joinpath("banana")

    fake_training = fake_training_fn(logdir=logdir)
    assert fake_training()
    trainer = fake_training.trainer
    restorer = Restorer(logdir=logdir)

    if isinstance(trainer, ClassifierTrainer):
        restorer: ClassifierRestorer = ClassifierRestorer(logdir=logdir)

        new_training: FakeClassifierTraining = fake_training_fn(_tmp_logdir)

        placeholder_model = new_training.model
        placeholder_opt = tf.keras.optimizers.Adam()

        # Ensure model have been built correctly
        x, _ = next(iter(new_training.dataset))
        placeholder_model(x)

        _test_convenience_model_restorer(
            restorer,
            restorer.restore_model,
            placeholder_model,
            trainer._model,
            trainer.ckpt_id_model,
            capsys,
        )
        _test_convenience_optimizer_restorer(
            restorer,
            restorer.restore_optimizer,
            placeholder_opt,
            trainer.ckpt_id_optimizer,
            capsys,
        )

    elif isinstance(trainer, AdversarialTrainer):
        restorer: AdversarialRestorer = AdversarialRestorer(logdir=logdir)

        new_training: FakeAdversarialTraining = fake_training_fn(_tmp_logdir)

        placeholder_g, placeholder_d = (
            new_training.generator,
            new_training.discriminator,
        )
        placeholder_optimizer_g, placeholder_optimizer_d = (
            tf.keras.optimizers.Adam(),
            tf.keras.optimizers.Adam(),
        )

        # Ensure that the ModelNotConstructedError is correctly triggered
        with pytest.raises(ModelNotConstructedError):
            _test_convenience_model_restorer(
                restorer,
                restorer.restore_generator,
                placeholder_g,
                trainer._generator,
                trainer.ckpt_id_generator,
                capsys,
            )

        with pytest.raises(ModelNotConstructedError):
            _test_convenience_model_restorer(
                restorer,
                restorer.restore_discriminator,
                placeholder_d,
                trainer._discriminator,
                trainer.ckpt_id_discriminator,
                capsys,
            )

        # Ensure models have been built correctly
        (x, _), z = next(iter(new_training.dataset))
        fake = placeholder_g(z)
        assert tf.reduce_all(tf.equal(fake.shape, x.shape))
        placeholder_d(x)

        _test_convenience_model_restorer(
            restorer,
            restorer.restore_generator,
            placeholder_g,
            trainer._generator,
            trainer.ckpt_id_generator,
            capsys,
        )
        _test_convenience_optimizer_restorer(
            restorer,
            restorer.restore_generator_optimizer,
            placeholder_optimizer_g,
            trainer.ckpt_id_optimizer_generator,
            capsys,
        )
        _test_convenience_model_restorer(
            restorer,
            restorer.restore_discriminator,
            placeholder_d,
            trainer._discriminator,
            trainer.ckpt_id_discriminator,
            capsys,
        )
        _test_convenience_optimizer_restorer(
            restorer,
            restorer.restore_discriminator_optimizer,
            placeholder_optimizer_d,
            trainer.ckpt_id_optimizer_discriminator,
            capsys,
        )


def test_failings(tmpdir):
    """Test the failing cases for the Restorers."""
    # Test Restorer fails on empty logdir
    with pytest.raises(FileNotFoundError):
        Restorer(Path(tmpdir + ("fuffa")))

    # Test Restorer fails on empty checkpoint dir
    with pytest.raises(FileNotFoundError):
        restorer = Restorer(tmpdir)
        restorer._restore_checkpoint(tf.train.Checkpoint())

    # Test failed placeholders validation
    with pytest.raises(TypeError):
        Restorer._validate_placeholder(
            placeholders=tf.keras.Model(), placeholder_type=tf.Variable,
        )
