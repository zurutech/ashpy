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

"""GAN metrics."""
import os
import types

import tensorflow as tf
import tensorflow_hub as hub

from ashpy.metrics import ClassifierMetric, Metric
from ashpy.modes import LogEvalMode


class DiscriminatorLoss(Metric):
    """The ash way of measuring the discriminator loss value."""

    def __init__(
        self, model_selection_operator=None, logdir=os.path.join(os.getcwd(), "log")
    ):
        """
        Args:
            model_selection_operator: The operation to be used when model_selection
                is on to compare the metrics. E.g.: operator.gt.
            logdir: The log dir.
        """

        super().__init__(
            name="d_loss",
            metric=tf.metrics.Mean(name="d_loss", dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        return self._metric.reset_states()

    def update_state(self, context):
        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy

            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )
            loss = context.discriminator_loss(
                context,
                fake=fake,
                real=real_x,
                condition=real_y,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )

            self._distribute_strategy.experimental_run_v2(
                lambda: self._metric.update_state(loss)
            )


class GeneratorLoss(Metric):
    """The ash way of measuring the generator loss value."""

    def __init__(
        self, model_selection_operator=None, logdir=os.path.join(os.getcwd(), "log")
    ):
        """
        Args:
            model_selection_operator: The operation to be used when model_selection
                is on to compare the metrics. E.g.: operator.gt.
            logdir: The log dir.
        """

        super().__init__(
            name="g_loss",
            metric=tf.metrics.Mean(name="g_loss", dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        return self._metric.reset_states()

    def update_state(self, context):
        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy
            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]

            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            loss = context.generator_loss(
                context,
                fake=fake,
                real=real_x,
                condition=real_y,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )

            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(loss)
            )


class EncoderLoss(Metric):
    """The ash way of measuring the generator loss value."""

    def __init__(
        self, model_selection_operator=None, logdir=os.path.join(os.getcwd(), "log")
    ):
        """
        Args:
            model_selection_operator: The operation to be used when model_selection
                is on to compare the metrics. E.g.: operator.gt.
            dtype: The type.
            logdir: The log dir.
        """

        super().__init__(
            name="e_loss",
            metric=tf.metrics.Mean(name="e_loss", dtype=tf.float32),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        return self._metric.reset_states()

    def update_state(self, context):
        for real_xy, noise in context.dataset:
            real_x, real_y = real_xy
            g_inputs = noise
            if len(context.generator_model.inputs) == 2:
                g_inputs = [noise, real_y]
            fake = context.generator_model(
                g_inputs, training=context.log_eval_mode == LogEvalMode.TRAIN
            )

            loss = context.encoder_loss(
                context,
                fake=fake,
                real=real_x,
                condition=real_y,
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )

            self._distribute_strategy.experimental_run(
                lambda: self._metric.update_state(loss)
            )


class InceptionScore(Metric):
    """The ash way of measuring the inception score."""

    def __init__(
        self,
        inception: tf.keras.Model,
        model_selection_operator=None,
        logdir=os.path.join(os.getcwd(), "log"),
    ):
        """
        This class is an implementation of the inception score for evaluating a GAN.

        Args:
            inception: Keras ineption model.
            model_selection_operator: An operator with which compare the results for
                the model selection, if needed.
            logdir: The log dir.
        """

        super().__init__(
            name="inception_score",
            metric=tf.metrics.Mean("inception_score"),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._incpt_model = inception
        if "softmax" not in self._incpt_model.layers[-1].name.lower():
            self._incpt_model = tf.keras.Sequential(
                [self._incpt_model, tf.keras.layers.Softmax()]
            )

    def update_state(self, context):

        # generate the images created with the context's generator
        generated_images = [
            context.generator_model(
                noise, training=context.log_eval_mode == LogEvalMode.TRAIN
            )
            for noise in context.noise_dataset
        ]

        rescaled_images = [
            ((generate_image * 0.5) + 0.5) for generate_image in generated_images
        ]

        # resize images to 299x299
        resized_images = [
            tf.image.resize(rescaled_image, (299, 299))
            for rescaled_image in rescaled_images
        ]

        try:
            resized_images[:] = [
                tf.image.grayscale_to_rgb(images) for images in resized_images
            ]
        except ValueError:
            # Images are already RGB
            pass

        # Instead of using multiple batches of 'batch_size' each (that causes OOM).
        # Unravel the dataset and then create small batches, each with 2 images at most.
        dataset = tf.unstack(tf.reshape(tf.stack(resized_images), (-1, 1, 299, 299, 3)))

        # calculate the inception score
        mean, _ = self.inception_score(dataset)

        # update the Mean metric created for this context
        # self._metric.update_state(mean)
        self._distribute_strategy.experimental_run(
            lambda: self._metric.update_state(mean)
        )

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        self._metric.reset_states()

    def inception_score(self, images, splits=10):
        """

        Args:
            images (ndarray): A list of ndarray of generated images of 299x299 of size.
            splits (int): The number of splits to be used during the inception score calculation.

        Returns:
             :obj:`None`.
        """

        tf.print("Computing inception score...")
        predictions = []
        for inp in images:
            pred = self._incpt_model(inp)
            predictions.append(pred)

        predictions = tf.concat(predictions, axis=0).numpy()
        scores = []
        for i in range(splits):
            part = predictions[
                (i * predictions.shape[0] // splits) : (
                    (i + 1) * predictions.shape[0] // splits
                ),
                :,
            ]
            kl = part * (
                tf.math.log(part)
                - tf.math.log(
                    tf.expand_dims(tf.math.reduce_mean(part, axis=0), axis=[0])
                )
            )
            kl = tf.math.reduce_mean(tf.math.reduce_sum(kl, axis=1))
            scores.append(tf.math.exp(kl))
        return tf.math.reduce_mean(scores).numpy(), tf.math.reduce_std(scores).numpy()

    @staticmethod
    def get_or_train_inception(
        dataset,
        name,
        num_classes,
        epochs,
        fine_tuning=False,
        loss_fn=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(1e-5),
        logdir=os.path.join(os.getcwd(), "log"),
    ):
        """
        Get or train and save the inception model.
        """
        os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
        model = tf.keras.Sequential(
            [
                hub.KerasLayer(
                    "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
                    output_shape=[2048],
                    trainable=fine_tuning,
                ),
                tf.keras.layers.Dense(512),
                tf.keras.layers.LeakyReLU(alpha=0.05),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        del os.environ["TFHUB_DOWNLOAD_PROGRESS"]
        step = tf.Variable(0, trainable=False, dtype=tf.int64)

        ckpt = tf.train.Checkpoint()
        ckpt.objects = []
        ckpt.objects.extend([model, step])
        logdir = logdir
        manager = tf.train.CheckpointManager(
            ckpt, os.path.join(logdir, "inception", name), max_to_keep=1
        )

        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            print(f"Restored checkpoint {manager.latest_checkpoint}.")
            return model

        print("Training the InceptionV3 model")

        def _train():
            def _step(features, labels):
                with tf.GradientTape() as tape:
                    logits = model(features)
                    loss = loss_fn(labels, logits)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                step.assign_add(1)
                tf.print(step, " loss value: ", loss)

            for epoch in range(epochs):
                for features, labels in dataset:
                    _step(features, labels)
                tf.print("epoch ", epoch, " completed")
                manager.save()

        _train()
        return model


class EncodingAccuracy(ClassifierMetric):
    """Measures the Generator and Encoder performance together, by classifying:
    G(E(x)), y using a pre-trained classified (on the dataset of x)."""

    def __init__(
        self,
        classifier: tf.keras.Model,
        model_selection_operator=None,
        logdir=os.path.join(os.getcwd(), "log"),
    ):
        """Measures the Generator and Encoder performance together, by classifying:
        G(E(x)), y using a pre-trained classified (on the dataset of x).

        Args:
            classifier: Keras inception model.
            model_selection_operator: An operator with which compare the results for
                the model selection, if needed.
            logdir: The log dir.
        """

        super().__init__(
            metric=tf.metrics.Accuracy("encoding_accuracy"),
            model_selection_operator=model_selection_operator,
            logdir=logdir,
        )

        self._classifer = classifier

    def update_state(self, context):
        inner_context = types.SimpleNamespace()
        inner_context.classifier_model = self._classifer
        inner_context.log_eval_mode = LogEvalMode.TEST
        # G(E(x)), y

        def _gen(xy, noise):
            # TODO: find a way to move generator_model
            # and encoder_model from GPU to CPU since tf.data.Dataset.map
            # requires every object allocated in CPU (perhaps)
            x, y = xy
            out = context.generator_model(
                context.encoder_model(
                    x, training=context.log_eval_mode == LogEvalMode.TRAIN
                ),
                training=context.log_eval_mode == LogEvalMode.TRAIN,
            )
            return out, y

        dataset = context.dataset.map(_gen)
        inner_context.dataset = dataset
        # Classify using the pre-trained classifier (self._classifier)
        # G(E(x)) and check the accuracy (with y)
        super().update_state(inner_context)

    def result(self):
        return self._metric.result().numpy()

    def reset_states(self):
        self._metric.reset_states()
