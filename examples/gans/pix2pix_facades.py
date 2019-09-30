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

"""
Pix2Pix on Facades Datasets dummy implementation.

Input Pipeline taken from: https://www.tensorflow.org/beta/tutorials/generative/pix2pix
"""
import os

import tensorflow as tf

from ashpy import LogEvalMode
from ashpy.losses.gan import (
    AdversarialLossType,
    Pix2PixLoss,
    get_adversarial_loss_discriminator,
)
from ashpy.models.convolutional.discriminators import PatchDiscriminator
from ashpy.models.convolutional.unet import FUNet
from ashpy.trainers.gan import AdversarialTrainer

_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"

PATH_TO_ZIP = tf.keras.utils.get_file("facades.tar.gz", origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(PATH_TO_ZIP), "facades/")

BUFFER_SIZE = 100
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    """Load the image from file path."""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    width = tf.shape(image)[1]

    width = width // 2
    real_image = image[:, :width, :]
    input_image = image[:, width:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    """Resize input_image and real_image to height x width."""
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image


def random_crop(input_image, real_image):
    """Random crop both input_image and real_image."""
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    """Normalize images in [-1, 1]."""
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image_train(image_file):
    """Load and process the image_file to be ready for the training."""
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


@tf.function
def random_jitter(input_image, real_image):
    """Apply random jitter to both input_image and real_image."""
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def main(
    kernel_size=5,
    learning_rate_d=2e-4,
    learning_rate_g=2e-4,
    g_input_res=IMG_WIDTH,
    g_min_res=1,
    g_initial_filters=64,
    g_filters_cap=512,
    use_dropout_encoder=False,
    use_dropout_decoder=True,
    d_target_res=32,
    d_initial_filters=64,
    d_filters_cap=512,
    use_dropout_discriminator=False,
    dataset_name="facades",
    resolution=256,
    epochs=100_000,
    dropout_prob=0.3,
    l1_loss_weight=100,
    gan_loss_weight=1,
    use_attention_d=False,
    use_attention_g=False,
    channels=3,
    gan_loss_type=AdversarialLossType.LSGAN,
):
    """Define Trainer and models."""
    generator = FUNet(
        input_res=g_input_res,
        min_res=g_min_res,
        kernel_size=kernel_size,
        initial_filters=g_initial_filters,
        filters_cap=g_filters_cap,
        channels=channels,  # color_to_label_tensor.shape[0],
        use_dropout_encoder=use_dropout_encoder,
        use_dropout_decoder=use_dropout_decoder,
        dropout_prob=dropout_prob,
        use_attention=use_attention_g,
    )
    discriminator = PatchDiscriminator(
        input_res=resolution,
        min_res=d_target_res,
        initial_filters=d_initial_filters,
        kernel_size=kernel_size,
        filters_cap=d_filters_cap,
        use_dropout=use_dropout_discriminator,
        dropout_prob=dropout_prob,
        use_attention=use_attention_d,
    )

    discriminator_loss = get_adversarial_loss_discriminator(gan_loss_type)()
    generator_loss = Pix2PixLoss(
        l1_loss_weight=l1_loss_weight,
        adversarial_loss_weight=gan_loss_weight,
        adversarial_loss_type=gan_loss_type,
    )

    metrics = []
    logdir = f'{"log"}/{dataset_name}/run2'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    trainer = AdversarialTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=tf.optimizers.Adam(learning_rate_g, beta_1=0.5),
        discriminator_optimizer=tf.optimizers.Adam(learning_rate_d, beta_1=0.5),
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        epochs=epochs,
        metrics=metrics,
        logdir=logdir,
        log_eval_mode=LogEvalMode.TEST,
    )

    train_dataset = tf.data.Dataset.list_files(PATH + "train/*.jpg")
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.map(load_image_train)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    train_dataset = train_dataset.map(lambda x, y: ((y, x), x))

    trainer(
        # generator_input,
        train_dataset
    )


if __name__ == "__main__":
    main()
