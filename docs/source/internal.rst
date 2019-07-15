AshPy Internals
###############

The two main concepts of AshPy internals are :py:class:`BaseContext <ashpy.contexts.base_context.BaseContext>` and :py:class:`Executor <ashpy.losses.executor.Executor>`.

Context
-------

A :py:class:`BaseContext <ashpy.contexts.base_context.BaseContext>` is an object that contains all the needed information. Here needed depends on the application.
In AshPy the :code:`Context` concept links a generic training loop with the loss function calculation and the model evaluation.
A :code:`Context` is a useful class in which all the models, metrics, dataset and mode of your network are set.
Passing the context around means that you can any time access to all what you need in order to perform any type of computation.

In AshPy we have (until now) three types of contexts:

- `Classifier Context`_
- `GAN Context`_
- `GANEncoder Context`_

Classifier Context
++++++++++++++++++

The :py:class:`ClassifierContext <ashpy.contexts.classifier.ClassifierContext>` is very simple, it contains only:

- classifier_model
- loss
- dataset
- metrics
- log_eval_mode
- global_step
- ckpt

In this way the loss function (:py:class:`Executor <ashpy.losses.executor.Executor>`) can use the context in order to get the model
and the needed information in order to correctly feed the model.

GAN Context
+++++++++++

The basic :py:class:`GANContext <ashpy.contexts.gan.GANContext>` is composed by:

- dataset
- generator_model
- discriminator_model
- generator_loss
- discriminator_loss
- metrics
- log_eval_mode
- global_step
- ckpt

As we can see we have all information needed to define our training and evaluation loop.

GANEncoder Context
++++++++++++++++++

The :py:class:`GANEncoderContext <ashpy.contexts.gan.GANEncoderContext>` extends the GANContext, contains all the
information of the base class plus:

- Encoder Model
- Encoder Loss

Executor
--------

The :py:class:`Executor <ashpy.losses.executor.Executor>` is the main concept behind the loss function implementation in AshPy.
An Executor is a class that helps in order to better generalize a training loop.
With an Executor you can construct, for example, a custom loss function and put every computation you need inside it.
You should define a :code:`call` function inside your class and decorate it with :code:`@Executor.reduce` header, if needed.

Inside the :code:`call` function you can take advantage of a context.

Executors can be summed up, subtracted and multiplied by scalars.

An executor takes also care of the distribution strategy by reducing appropriately the loss (see
`Tensorflow Guide`__).

An Executor Example
*******************

In this example we will see the implementation of the Generator Binary CrossEntropy loss.

The :code:`__init__` method is straightforward, we need only to instantiate :py:class:`tf.losses.BinaryCrossentropy` object and then we pass it to our parent:

.. code-block:: python

    class GeneratorBCE(GANExecutor):

        def __init__(self, from_logits=True):
            self.name = "GeneratorBCE"
            # call super passing the BinaryCrossentropy as function
            super().__init__(tf.losses.BinaryCrossentropy(from_logits=from_logits))

Then we need to implement the call function respecting the signature:

.. code-block:: python

        def call(self, context, *, fake, condition, training, **kwargs):

            # we need a function that gives us the correct inputs given the discriminator model
            fake_inputs = self.get_discriminator_inputs(
                context=context, fake_or_real=fake, condition=condition, training=training
            )

            # get the discriminator predictions from the discriminator model
            d_fake = context.discriminator_model(fake_inputs, training=training)

            # get the target prediction for the generator
            value = self._fn(tf.ones_like(d_fake), d_fake)

            # mean everything
            return tf.reduce_mean(value)

The function :py:func:`get_discriminator_inputs` returns the correct discriminator inputs using the context.
The discriminator input can be the output of the generator (unconditioned case) or the output of the generator together
with the condition (conditioned case).

The the :py:func:`call` uses the discriminator model inside the context in order to obtain the output of the
discriminator when evaluated in the `fake_inputs`.

After that the :py:func:`self._fn` (BinaryCrossentropy) is used to get the value of the loss. This loss is then averaged.

In this way the executor computes correctly the loss function.

This is ok if we do not want use our code in a distribution strategy.

If we want to use our executor in a distribution strategy the only modifications are:

.. code-block:: python

        @Executor.reduce_loss
        def call(self, context, *, fake, condition, training, **kwargs):

            # we need a function that gives us the correct inputs given the discriminator model
            fake_inputs = self.get_discriminator_inputs(
                context=context, fake_or_real=fake, condition=condition, training=training
            )

            # get the discriminator predictions from the discriminator model
            d_fake = context.discriminator_model(fake_inputs, training=training)

            # get the target prediction for the generator
            value = self._fn(tf.ones_like(d_fake), d_fake)

            # mean only over the axis 1
            return tf.reduce_mean(value, axis=1)

The important things are:

- :code:`Executor.reduce_loss` decoration: uses the Executor decorator in order to correctly reduce the loss
- :code:`tf.reduce_mean(value, axis=1)` (last line), we perform only the mean over the axis 1. The output of the `call` function
should be a :py:class:`tf.Tensor` with shape (N, 1) or (N,). This is because the decorator performs the mean over the axis 0.

.. _tf_guide: https://www.tensorflow.org/beta/guide/distribute_strategy#using_tfdistributestrategy_with_custom_training_loops
__ tf_guide_