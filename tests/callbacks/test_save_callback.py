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

"""Test Save Callback."""
import os
from typing import Tuple

import pytest

from ashpy.callbacks import SaveCallback, SaveFormat, SaveSubFormat
from ashpy.models.gans import ConvDiscriminator, ConvGenerator
from tests.utils.fake_training_loop import fake_training_loop

COMPATIBLE_FORMAT_AND_SUB_FORMAT = [
    (SaveFormat.WEIGHTS, SaveSubFormat.TF),
    (SaveFormat.WEIGHTS, SaveSubFormat.H5),
    (SaveFormat.MODEL, SaveSubFormat.TF),
]

INCOMPATIBLE_FORMAT_AND_SUB_FORMAT = [(SaveFormat.MODEL, SaveSubFormat.H5)]


@pytest.mark.parametrize("save_format_and_sub_format", COMPATIBLE_FORMAT_AND_SUB_FORMAT)
def test_save_callback_compatible(
    adversarial_logdir: str,
    save_format_and_sub_format: Tuple[SaveFormat, SaveSubFormat],
    save_dir: str,
):
    """Test the integration between callbacks and trainer."""
    save_format, save_sub_format = save_format_and_sub_format
    _test_save_callback_helper(
        adversarial_logdir, save_format, save_sub_format, save_dir
    )

    save_dirs = os.listdir(save_dir)
    # 2 folders: generator and discriminator
    assert len(save_dirs) == 2

    for model_dir in save_dirs:
        assert save_format.name() in [
            x.split(os.path.sep)[-1]
            for x in os.listdir(os.path.join(save_dir, model_dir))
        ]


@pytest.mark.parametrize(
    "save_format_and_sub_format", INCOMPATIBLE_FORMAT_AND_SUB_FORMAT
)
def test_save_callback_incompatible(
    adversarial_logdir: str,
    save_format_and_sub_format: Tuple[SaveFormat, SaveSubFormat],
    save_dir: str,
):
    """Test the integration between callbacks and trainer."""
    save_format, save_sub_format = save_format_and_sub_format

    with pytest.raises(NotImplementedError):
        _test_save_callback_helper(
            adversarial_logdir, save_format, save_sub_format, save_dir
        )

    # assert no folder has been created
    assert not os.path.exists(save_dir)


def _test_save_callback_helper(
    adversarial_logdir, save_format, save_sub_format, save_dir
):
    image_resolution = (28, 28)
    layer_spec_input_res = (7, 7)
    layer_spec_target_res = (7, 7)
    kernel_size = 5
    channels = 1

    # model definition
    generator = ConvGenerator(
        layer_spec_input_res=layer_spec_input_res,
        layer_spec_target_res=image_resolution,
        kernel_size=kernel_size,
        initial_filters=32,
        filters_cap=16,
        channels=channels,
    )

    discriminator = ConvDiscriminator(
        layer_spec_input_res=image_resolution,
        layer_spec_target_res=layer_spec_target_res,
        kernel_size=kernel_size,
        initial_filters=16,
        filters_cap=32,
        output_shape=1,
    )

    callbacks = [
        SaveCallback(
            models=[generator, discriminator],
            save_dir=save_dir,
            verbose=1,
            save_format=save_format,
            save_sub_format=save_sub_format,
        )
    ]

    fake_training_loop(
        adversarial_logdir,
        callbacks=callbacks,
        generator=generator,
        discriminator=discriminator,
    )


def test_save_callback_type_error(
    save_dir: str,
):
    """Test that the SaveCallback raises a TypeError.

    Test that the SaveCallback raises a TypeError when wrong save_format
    or save sub-format is passed.
    """
    with pytest.raises(TypeError):
        callbacks = [
            SaveCallback(
                models=[],
                save_dir=save_dir,
                verbose=1,
                save_format="save_format",
                save_sub_format=SaveSubFormat.TF,
            )
        ]

    with pytest.raises(TypeError):
        callbacks = [
            SaveCallback(
                models=[],
                save_dir=save_dir,
                verbose=1,
                save_format=SaveFormat.WEIGHTS,
                save_sub_format="sub-format",
            )
        ]
