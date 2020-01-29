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


def test_generate_human_ckpt_dict(fake_training, tmpdir):
    """
    Test that the generation of the human readable map of the ckpt_dict works.

    TODO: improve the test.
    """
    logdir = Path(tmpdir)
    training_loop, loop_args, metrics = fake_training
    training_completed, trainer = training_loop(
        logdir=logdir, metrics=metrics, **loop_args
    )
    assert training_completed
    assert trainer._checkpoint_map
    assert (Path(trainer._ckpts_dir) / "checkpoint_map.json").exists()
    metrics: List[ashpy.metrics.Metric] = trainer._metrics
    for metric in metrics:
        assert metric.best_folder / "ckpts" / "checkpoint_map.json"
