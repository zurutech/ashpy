Getting Started
###############


Datasets
========

AshPy supports :py:class:`tf.data.Dataset` format.

We highly encourage you to use `Tensorflow Datasets`__ to manage and use your datasets in an handy way.

.. code-block:: bash

    pip install tfds-nightly

Classification
++++++++++++++

In order to create a dataset for classification:

.. code-block:: python

    import tensorflow_datasets as tfds

    from ashpy.trainers import ClassifierTrainer

    def extract_fn(example):
        return example["image"], example["label"]

    def main():
        ds_train, ds_validation = tfds.load(name="mnist", split=["train", "validation"])

        # build the input pipeline
        ds_train = ds_train.batch(BATCH_SIZE).prefetch(1)
        ds_train = ds_train.map(extract_fn)

        # same for validation
        ...

        # define model, loss, optimizer
        ...

        # define the classifier trainer
        trainer = ClassifierTrainer(model, optimizer, loss, epochs, metrics, logdir=logdir)

        # train
        trainer.train(ds_train, ds_validation)


GANs
++++

In order to create a datasets for a (Conditional) GANs:

.. code-block:: python

    import tensorflow_datasets as tfds

    from ashpy.trainers import AdversarialTrainer

    def extract_fn(example):
        # the ashpy input must be (real, condition), condition
        return (example["image"], example["label"]), example["label"]

    def main():
        ds_train = tfds.load(name="mnist", split="train")

        # build the input pipeline
        ds_train = ds_train.batch(BATCH_SIZE).prefetch(1)
        ds_train = ds_train.map(extract_fn)

        # define models, losses, optimizers
        ...

        # define the adversarial trainer
        trainer = AdversarialTrainer(generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss,
            discriminator_loss,
            epochs,
            metrics,
            logdir,
        )

        # train
        trainer.train(ds_train)


.. _tfds: https://github.com/tensorflow/datasets
__ tfds_

Models
======

AshPy supports `Keras`__ models as inputs.
You can use an AshPy predefined model or you can implement your own model.

Using an AshPy model
++++++++++++++++++++

.. code-block:: python

    import tensorflow_datasets as tfds

    from ashpy.trainers import ClassifierTrainer
    from ashpy.models import UNet

    def main():

        # create the dataset and the input pipeline

        # define models, loss, optimizer
        model = UNet(
            input_res,
            min_res,
            kernel_size,
            initial_filters,
            filters_cap,
            channels,
            use_dropout_encoder,
            use_dropout_decoder,
            dropout_prob,
            use_attention,
        )

        # define the classifier trainer
        trainer = AdversarialTrainer(generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss,
            discriminator_loss,
            epochs,
            metrics,
            logdir,
        )

        # train
        trainer.train(ds_train)


Creating a Model
++++++++++++++++

It's very easy to create a simple model, since AshPy's models are Keras' models.


.. code-block:: python

    from ashpy.layers import Attention, InstanceNormalization

    def downsample(
        filters,
        apply_normalization=True,
        attention=False,
        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
        size=3,
    ):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=not apply_normalization,
        )
    )

    if apply_normalization:
        result.add(InstanceNormalization())

    result.add(activation)

    if attention:
        result.add(Attention(filters))

    return result


    def upsample(
        filters,
        apply_dropout=False,
        apply_normalization=True,
        attention=False,
        activation=tf.keras.layers.ReLU(),
        size=3,
    ):
        initializer = tf.random_normal_initializer(0.0, 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        result.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))

        result.add(
            tf.keras.layers.Conv2D(
                filters,
                size,
                strides=1,
                padding="valid",
                kernel_initializer=initializer,
                use_bias=False,
            )
        )

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        if apply_normalization:
            result.add(Normalizer())

        result.add(activation)

        if attention:
            result.add(Attention(filters))

        return result


    def Generator(attention, output_channels=3):
        down_stack = [
            downsample(32, apply_normalization=False),  # 256
            downsample(32),  # 128
            downsample(64, attention=attention),  # 64
            downsample(64),  # 32
            downsample(64),  # 16
            downsample(128),  # 8
            downsample(128),  # 4
            downsample(256),  # 2
            downsample(512, apply_normalization=False),  # 1
        ]

        up_stack = [
            upsample(256, apply_dropout=True),  # 2
            upsample(128, apply_dropout=True),  # 4
            upsample(128, apply_dropout=True),  # 8
            upsample(64),  # 16
            upsample(64),  # 32
            upsample(64, attention=attention),  # 64
            upsample(32),  # 128
            upsample(32),  # 256
            upsample(32),  # 512
        ]

        inputs = tf.keras.layers.Input(shape=[None, None, 1])
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        last = upsample(
            output_channels,
            activation=tf.keras.layers.Activation(tf.nn.tanh),
            apply_normalization=False,
        )

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)



In this way we have created a new model to be used inside AshPy.

Inheriting from ashpy.models.Conv2DInterface
++++++++++++++++++++++++++++++++++++++++++++

The third possibility you have to create a new model is to inherit from the :py:class:`ashpy.models.convolutional.interfaces.Conv2DInterface`.

This class offers the basic methods to implement in a simple way a new model.

.. _keras: https://www.tensorflow.org/guide/keras
__ keras_

Creating a new Trainer
======================

AshPy has different generics trainers.
Trainers implement the basic training loop together with distribution strategy management and logging.
By now the only distribution strategy handled is the :py:class:`tf.distribute.MirroredStrategy`.

----

Complete Examples
=================

Classifier
++++++++++

.. literalinclude:: ../../examples/classifier.py
   :language: python
   :linenos:

GANs
++++

BiGAN
-----

.. literalinclude:: ../../examples/gans/bigan.py
   :language: python
   :linenos:

MNIST
-----

.. literalinclude:: ../../examples/gans/mnist.py
   :language: python
   :linenos:

Facades (Pix2Pix)
-----------------

.. literalinclude:: ../../examples/gans/pix2pix_facades.py
   :language: python
   :linenos:
