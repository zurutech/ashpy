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

"""Tests for :mod:`ashpy.trainers`."""

from pathlib import Path
from typing import List

import ashpy
import pytest
import tensorflow as tf
from ashpy.losses import ClassifierLoss
from ashpy.trainers import AdversarialTrainer, ClassifierTrainer

from tests.test_restorers import _check_models_weights
from tests.utils.fake_training_loop import (
    FakeAdversarialTraining,
    FakeClassifierTraining,
    FakeTraining,
)


def test_correct_trainer_restoration_on_restart(fake_training_fn, tmpdir):
    logdir = Path(tmpdir)

    fake_training: FakeTraining = fake_training_fn(logdir=logdir)
    assert fake_training()

    if isinstance(fake_training, FakeClassifierTraining):
        assert isinstance(fake_training.trainer, ClassifierTrainer)
        trained_model = fake_training.trainer._model

        new_training: FakeClassifierTraining = fake_training_fn(logdir=logdir)
        new_training.trainer._build_and_restore_models(new_training.dataset)
        restored_model = new_training.trainer._model

        _check_models_weights(trained_model, restored_model)

    if isinstance(fake_training, FakeAdversarialTraining):
        assert isinstance(fake_training.trainer, AdversarialTrainer)
        trained_g = fake_training.trainer._generator
        trained_d = fake_training.trainer._discriminator

        new_training: FakeAdversarialTraining = fake_training_fn(logdir=logdir)
        new_training.trainer._build_and_restore_models(new_training.dataset)
        restored_g = new_training.trainer._generator
        restored_d = new_training.trainer._discriminator
        _check_models_weights(trained_g, restored_g)
        _check_models_weights(trained_d, restored_d)


def test_generate_human_ckpt_dict(fake_training_fn, tmpdir):
    """
    Test that the generation of the human readable map of the ckpt_dict works.

    TODO: improve the test.
    """
    logdir = Path(tmpdir)
    fake_training = fake_training_fn(logdir=logdir)
    assert fake_training()

    trainer = fake_training.trainer

    assert trainer._checkpoint_map
    assert (Path(trainer._ckpts_dir) / "checkpoint_map.json").exists()
    metrics: List[ashpy.metrics.Metric] = trainer._metrics
    for metric in metrics:
        assert metric.best_folder / "ckpts" / "checkpoint_map.json"


def test_loss_names_collision_classifier(tmpdir):
    """
    Test that an exception is correctly raised when two losses have the same name for the
    classifier trainer.

    WHEN two or more losses passed to a trainer have the same name
        THEN raise a ValueError
    """
    loss = ClassifierLoss(
        fn=tf.keras.losses.BinaryCrossentropy(), name="bce"
    ) + ClassifierLoss(fn=tf.keras.losses.BinaryCrossentropy(), name="bce")

    with pytest.raises(ValueError):
        FakeClassifierTraining(logdir=tmpdir, loss=loss)


def test_loss_names_collision_adversarial(tmpdir):
    """
    Test that an exception is correctly raised when two losses have the same name adversarial
    trainer.

    WHEN two or more losses passed to a trainer have the same name
        THEN raise a ValueError
    """
    generator_loss = ashpy.losses.gan.GeneratorL1(name="loss")
    discriminator_loss = ashpy.losses.gan.DiscriminatorLSGAN(name="loss")

    with pytest.raises(ValueError):
        FakeAdversarialTraining(
            logdir=tmpdir,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
        )


def test_loss_names_collision_sum_executor_adversarial(tmpdir):
    """
    Test that an exception is correctly raised when two losses have the same name adversarial
    trainer.

    WHEN two or more losses passed to a trainer have the same name
        THEN raise a ValueError
    """
    generator_loss = ashpy.losses.gan.Pix2PixLoss(name="loss")
    discriminator_loss = ashpy.losses.gan.DiscriminatorLSGAN(name="loss")

    with pytest.raises(ValueError):
        FakeAdversarialTraining(
            logdir=tmpdir,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
        )
