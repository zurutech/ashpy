<div align="center">
    <img src="https://blog.zuru.tech/images/ashpy/logo_lq.png" />
</div>

# AshPy

![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Python - Version](https://img.shields.io/pypi/pyversions/ashpy.svg)
![PyPy - Version](https://badge.fury.io/py/ashpy.svg)
![PyPI - License](https://img.shields.io/pypi/l/ashpy.svg)
![Ashpy - Badge](https://img.shields.io/badge/package-ashpy-brightgreen.svg)
[![codecov](https://codecov.io/gh/zurutech/ashpy/branch/master/graph/badge.svg)](https://codecov.io/gh/zurutech/ashpy)
[![Build Status](https://travis-ci.org/zurutech/ashpy.svg?branch=master)](https://travis-ci.org/zurutech/ashpy)
[![Documentation Status](https://readthedocs.org/projects/ashpy/badge/?version=latest)](https://ashpy.readthedocs.io/en/latest/?badge=latest)
[![Black - Badge](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![CodeFactor](https://www.codefactor.io/repository/github/zurutech/ashpy/badge)](https://www.codefactor.io/repository/github/zurutech/ashpy)

AshPy is a TensorFlow 2.1 library for (**distributed**) training, evaluation, model selection, and fast prototyping.
It is designed to ease the burden of setting up all the nuances of the architectures built to train complex custom deep learning models.

[Quick Example](#quick-example) | [Features](#features) | [Set Up](#set-up) | [Usage](#usage) | [Dataset Output Format](#dataset-output-format) | [Test](#test)

## Quick Example

```python
# define a distribution strategy
strategy = tf.distribute.MirroredStrategy()

# work inside the scope of the created strategy
with strategy.scope():

    # get the MNIST dataset
    train, validation = tf.keras.datasets.mnist.load_data()

    # process data if needed
    def process(images, labels):
        data_images = tf.data.Dataset.from_tensor_slices((images)).map(
            lambda x: tf.reshape(x, (28 * 28,))
        )
        data_images = data_images.map(
            lambda x: tf.image.convert_image_dtype(x, tf.float32)
        )
        data_labels = tf.data.Dataset.from_tensor_slices((labels))
        dataset = tf.data.Dataset.zip((data_images, data_labels))
        dataset = dataset.batch(1024 * 1)
        return dataset

    # apply the process function to the data
    train, validation = (
        process(train[0], train[1]),
        process(validation[0], validation[1]),
    )

    # create the model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(10),
        ]
    )

    # define the optimizer
    optimizer = tf.optimizers.Adam(1e-3)

    # the loss is provided by the AshPy library
    loss = ClassifierLoss(tf.losses.SparseCategoricalCrossentropy(from_logits=True))
    logdir = "testlog"
    epochs = 10

    # the metrics are provided by the AshPy library
    # and every metric with model_selection_operator != None performs
    # model selection, saving the best model in a different folder per metric.
    metrics = [
        ClassifierMetric(
            tf.metrics.Accuracy(), model_selection_operator=operator.gt
        ),
        ClassifierMetric(
            tf.metrics.BinaryAccuracy(), model_selection_operator=operator.gt
        ),
    ]

    # define the AshPy trainer
    trainer = ClassifierTrainer(
        model, optimizer, loss, epochs, metrics, logdir=logdir
    )

    # run the training process
    trainer(train, validation)
```

## Features

AshPy is a library designed to ease the burden of setting up all the nuances of the architectures built to train complex custom deep learning models. It provides both fully convolutional and fully connected models such as:

- autoencoder
- decoder
- encoder

and a fully convolutional:

- unet

Moreover, it provides already prepared trainers for a classifier model and GAN networks. In particular, in regards of the latter, it offers a basic GAN architecture with a Generator-Discriminator structure and an enhanced GAN architecture version made up of a Encoder-Generator-Discriminator structure.

---

AshPy it is developed around the concepts of _Executor_, _Context_, _Metric_, and _Strategies_ that represents its foundations.

**Executor** An Executor is a class that helps to better generalize a training loop. With an Executor you can construct, for example, a custom loss function and put whatever computation you need inside it. You should define a `call` function inside your class and decorate it with `@Executor.reduce` header. Inside the `call` function you can take advantage of a context.

**Context** A Context is a useful class in which all the models, metrics, dataset and mode of your network are set. Passing the context around means that you can any time access to all what you need in order to performs any type of computation.

**Metric** A Metric is a class from which you can inherit to create your custom metric that can automatically keep track of the best performance of the model during training and, automatically save the best one doing what is called the *model selection*.

**Strategies** If you want to distribute your training across multiple GPUs, there is the `tf.distribute.Strategy` TensorFlow API with which you can distribute your models and training code with minimal code changes. AshPy implements this type of strategies internally and will check everything for you to apply the distribution strategy correctly. All you need to do is as simple as doing the following:

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    generator = ConvGenerator(
        layer_spec_input_res=(7, 7),
        layer_spec_target_res=(28, 28),
        kernel_size=(5, 5),
        initial_filters=256,
        filters_cap=16,
        channels=1,
    )
    # rest of the code
    # with trainer definition and so on
```

i.e., create the strategy and put the rest of the code inside its scope.

In general AshPy aims to:

- Rapid model prototyping
- Enforcement of best practices & API consistency
- Remove duplicated and boilerplate code
- General usability by new project

**NOTE:** We invite you to read the full documentation on [the official website](https://ashpy.zurutech.io/).

The following README aims to help you understand what you need to do to setup AshPy on your system and, with some examples, what you need to do to setup a complete training of your network. Moreover, it will explain some fundamental modules you need to understand to fully exploit the potential of the library.

## Set up

### Pip install
```bash
pip install ashpy
```

### Source install

Clone this repo, go inside the downloaded folder and install with:
```bash
pip install -e .
```

## Usage

Let's quickly start with some examples.

### Classifier

Let's say we want to train a classifier.

```python
import operator
import tensorflow as tf
from ashpy.metrics import ClassifierMetric
from ashpy.trainers.classifier import ClassifierTrainer
from ashpy.losses.classifier import ClassifierLoss

def toy_dataset():
    inputs = tf.expand_dims(tf.range(1, 1000.0), -1)
    labels = tf.expand_dims([1 if tf.equal(tf.math.mod(tf.squeeze(i), 2), 0) else 0 for i in inputs], -1)
    return tf.data.Dataset.from_tensor_slices((inputs,labels)).shuffle(10).batch(2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(2)
])

optimizer = tf.optimizers.Adam(1e-3)
loss = ClassifierLoss(tf.losses.SparseCategoricalCrossentropy(from_logits=True))
logdir = "testlog"
epochs = 2

metrics = [
    ClassifierMetric(tf.metrics.Accuracy(), model_selection_operator=operator.gt),
    ClassifierMetric(tf.metrics.BinaryAccuracy(), model_selection_operator=operator.gt),
]

trainer = ClassifierTrainer(model, optimizer, loss, epochs, metrics, logdir=logdir)

train, validation = toy_dataset(), toy_dataset()
trainer(train, validation)
```

Skipping the `toy_dataset()` function that creates a toy dataset, we'll give a look to the code step by step.

So, first of all we define a model and its optimizer. Here, the model is a very simple sequential Keras model defined as:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(2)
])

optimizer = tf.optimizers.Adam(1e-3)
```

Then we define the loss:

```python
loss = ClassifierLoss(tf.losses.SparseCategoricalCrossentropy(from_logits=True))
```

The `ClassifierLoss` loss defined above it is defined using an internal class called "`Executor`". The Executor is a class that let you define, alongside with a desired loss, the function that you want to use to "evaluate" that loss with all the needed parameters.

This works in conjunction with the following line (we will speak about the "_metrics_" and the other few definition lines in a minute):

```python
trainer = ClassifierTrainer(model, optimizer, loss, epochs, metrics, logdir=logdir)
```

where a `ClassifierTrainer` is an object designed to run a specific training procedure adjusted, in this case, for a classifier.

The arguments of this function are the model, the optimizer, the loss, the number of epochs, the metrics and the logdir. We have already seen the definition of the model, the optimizer and of the loss. The definition of epochs, metrics and logdir happens here:

```python
logdir = "testlog"
epochs = 2

metrics = [
    ClassifierMetric(tf.metrics.Accuracy(), model_selection_operator=operator.gt),
    ClassifierMetric(
    tf.metrics.BinaryAccuracy(),model_selection_operator=operator.gt),
]
```

What we need to underline here is the definition of the metrics because as you can see they are defined through the use of specific classes: `ClassifierMetric`. As for the `ClassifierTrainer`, the `ClassifierMetric` it is a specified designed class for the Classifier. If you want to create a different metric you should inheriting from the Metric class provided by the Ash library. This kind of Metrics are useful because you can indicate a processing function to apply on predictions (e.g., tf.argmax) and an operator (e.g., operator.gt is the "greater than" operator) if you desire to activate the model selection during the training process based on that particular metric.

Finally, once the datasets has been set, you can start the training procedure calling the trainer object:

```python
train, validation = toy_dataset(), toy_dataset()
trainer(train, validation)
```

## GAN - Generative Adversarial Network

AshPy is equipped with two types of GAN network architectures:

- A plain GAN network with the classic structure Generator - Discriminator.
- A more elaborated GAN network architecture with the classic Generator - Discriminator structure plus an Encoder model (BiGAN like).

As for the previous classifier training example, let's see for first a simple example of an entire "toy" code, regarding a simple plain GAN. At the end we will briefly touch upon the differences with the GAN network with the Encoder.

```python
import operator
import tensorflow as tf
from ashpy.models.gans import ConvGenerator, ConvDiscriminator
from ashpy.metrics import InceptionScore
from ashpy.losses.gan import DiscriminatorMinMax, GeneratorBCE

generator = ConvGenerator(
    layer_spec_input_res=(7, 7),
    layer_spec_target_res=(28, 28),
    kernel_size=(5, 5),
    initial_filters=32,
    filters_cap=16,
    channels=1,
)

discriminator = ConvDiscriminator(
    layer_spec_input_res=(28, 28),
    layer_spec_target_res=(7, 7),
    kernel_size=(5, 5),
    initial_filters=16,
    filters_cap=32,
    output_shape=1,
)

# Losses
generator_bce = GeneratorBCE()
minmax = DiscriminatorMinMax()

# Real data
batch_size = 2
mnist_x, mnist_y = tf.zeros((100,28,28)), tf.zeros((100,))

# Trainer
epochs = 2
logdir = "testlog/adversarial"

metrics = [
    InceptionScore(
        # Fake inception model
        ConvDiscriminator(
            layer_spec_input_res=(299, 299),
            layer_spec_target_res=(7, 7),
            kernel_size=(5, 5),
            initial_filters=16,
            filters_cap=32,
            output_shape=10,
        ),
        model_selection_operator=operator.gt,
        logdir=logdir,
    )
]

trainer = AdversarialTrainer(
    generator,
    discriminator,
    tf.optimizers.Adam(1e-4),
    tf.optimizers.Adam(1e-4),
    generator_bce,
    minmax,
    epochs,
    metrics,
    logdir,
)

# Dataset
noise_dataset = tf.data.Dataset.from_tensors(0).repeat().map(
    lambda _: tf.random.normal(shape=(100,), dtype=tf.float32, mean=0.0, stddev=1)
).batch(batch_size).prefetch(1)

# take only 2 samples to speed up tests
real_data = tf.data.Dataset.from_tensor_slices(
        (tf.expand_dims(mnist_x, -1), tf.expand_dims(mnist_y, -1))
    ).take(batch_size).batch(batch_size).prefetch(1)

# Add noise in the same dataset, just by mapping.
# The return type of the dataset must be: tuple(tuple(a,b), noise)
dataset = real_data.map(lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, 100))))

trainer(dataset)
```

First we define the generator and discriminator of the GAN architecture:

```python
generator = ConvGenerator(
    layer_spec_input_res=(7, 7),
    layer_spec_target_res=(28, 28),
    kernel_size=(5, 5),
    initial_filters=32,
    filters_cap=16,
    channels=1,
)

discriminator = ConvDiscriminator(
    layer_spec_input_res=(28, 28),
    layer_spec_target_res=(7, 7),
    kernel_size=(5, 5),
    initial_filters=16,
    filters_cap=32,
    output_shape=1,
)
```

and then we define the losses:

```python
# Losses
generator_bce = GeneratorBCE()
minmax = DiscriminatorMinMax()
```

where `GeneratorBCE()` and `DiscriminatorMinMax()` are the losses defined inheriting `Executor`. Again, as we have seen in the previous classifier example, you can customize this type (the ones inheriting from the `Executor`) of losses.

The metrics are defined as follow:

```python
metrics = [
    InceptionScore(
    # Fake inception model
        ConvDiscriminator(
        layer_spec_input_res=(299, 299),
        layer_spec_target_res=(7, 7),
        kernel_size=(5, 5),
        initial_filters=16,
        filters_cap=32,
        output_shape=10,
        ),
        model_selection_operator=operator.gt,
        logdir=logdir,
    )
]
```

and in particular here we have the InceptionScore metric constructed on the fly with the ConvDiscriminator class provided by AshPy.

Finally, the actual trainer is constructed and then called:

```python
trainer = AdversarialTrainer(
    generator,
    discriminator,
    tf.optimizers.Adam(1e-4),
    tf.optimizers.Adam(1e-4),
    generator_bce,
    minmax,
    epochs,
    metrics,
    logdir,
)
```

```python
trainer(dataset)
```

The main difference with a GAN architecture with an Encoder is that we would have the encoder loss:

```python
encoder_bce = EncoderBCE()
```

an encoder accuracy metric:

```python
metrics = [EncodingAccuracy(classifier, model_selection_operator=operator.gt, logdir=logdir)]
```

and an EncoderTrainer:

```python
trainer = EncoderTrainer(
    generator,
    discriminator,
    encoder,
    tf.optimizers.Adam(1e-4),
    tf.optimizers.Adam(1e-5),
    tf.optimizers.Adam(1e-6),
    generator_bce,
    minmax,
    encoder_bce,
    epochs,
    metrics=metrics,
    logdir=logdir,
)
```

Note that the `EncoderTrainer` indicates a trainer of a GAN network with an Encoder and not a trainer of an Encoder itself.

## Dataset Output Format

In order to standardize the GAN training, AshPy requires the input dataset to be in a common format. In particular, the dataset return type must always be in the format showed below, where the fist element of the tuple is the discriminator input, and the second is the generator input.

```
tuple(tuple(a,b), noise)
```

Where `a` is the input sample, `b` is the label/condition (if any, otherwise fill it with `0`), and `noise` is the latent vector of input.

To train Pix2Pix-like architecture, that have no `noise` as ConvGenerator input, just return the values in thee format `(tuple(a,b), b)` since the condition is the generator input.

## Test
In order to run the tests (with the doctests), linting and docs generation simply use `tox`.

```bash
tox
```
