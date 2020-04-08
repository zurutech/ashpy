Advanced AshPy
##############

Custom Metrics
==============

AshPy Trainers can accept metrics that they will use for both logging and automatic model
selection.

Implementing a custom Metric in AshPy can be done via two approach:

1. Your metrics is already available as a :class:`tf.keras.metrics.Metric` and you want to use it as is.
2. You need to write the implementation of the Metric from scratch or you need to alter the default behavior we provide for AshPy Metrics.

Wrapping Keras Metrics
----------------------

In case number (1) what you want to do is to search for one of the Metrics provided by AshPy
and use it as a wrapper around the one you wish to use.

.. note::
    Passing an :mod:`operator` funciton to the AshPy Metric will enable model selection using the
    metric value.

The example below shows how to implement the Precision metric for an
:class:`ashpy.trainers.ClassifierTrainer`.

.. code-block:: python

    import operator

    from ashpy.metrics import ClassifierMetric
    from ashpy.trainers import ClassifierTrainer
    from tensorflow.keras.metrics import Precision

    precision = ClassifierMetric(
        metric=tf.keras.metrics.Precision(),
        model_selection_operator=operator.gt,
        logdir=Path().cwd() / "log",
    )

    trainer = ClassifierTrainer(
        ...
        metrics = [precision]
        ...
    )

You can apply this technique to any object derived and behaving as a
:class:`tf.keras.metrics.Metric` (i.e. the Metrics present in `TensorFlow Addons`_)


Creating your own Metric
------------------------

As an example of a custom Metric we present the analysis of the :class:`ashpy.metrics.classifier.ClassifierLoss`.

.. code-block:: python

    class ClassifierLoss(Metric):
        """A handy way to measure the classification loss."""

        def __init__(
            self,
            name: str = "loss",
            model_selection_operator: Callable = None,
            logdir: Union[Path, str] = Path().cwd() / "log",
        ) -> None:
            """
            Initialize the Metric.
            Args:
                name (str): Name of the metric.
                model_selection_operator (:py:obj:`typing.Callable`): The operation that will
                    be used when `model_selection` is triggered to compare the metrics,
                    used by the `update_state`.
                    Any :py:obj:`typing.Callable` behaving like an :py:mod:`operator` is accepted.
                    .. note::
                        Model selection is done ONLY if an operator is specified here.
                logdir (str): Path to the log dir, defaults to a `log` folder in the current
                    directory.
            """
            super().__init__(
                name=name,
                metric=tf.metrics.Mean(name=name, dtype=tf.float32),
                model_selection_operator=model_selection_operator,
                logdir=logdir,
            )

        def update_state(self, context: ClassifierContext) -> None:
            """
            Update the internal state of the metric, using the information from the context object.
            Args:
                context (:py:class:`ashpy.contexts.ClassifierContext`): An AshPy Context
                    holding all the information the Metric needs.
            """
            updater = lambda value: lambda: self._metric.update_state(value)
            for features, labels in context.dataset:
                loss = context.loss(
                    context,
                    features=features,
                    labels=labels,
                    training=context.log_eval_mode == LogEvalMode.TRAIN,
                )
                self._distribute_strategy.experimental_run_v2(updater(loss))


* Each custom Metric should always inherit from :class:`ashpy.metrics.metric.Metric`.
* We advise that each custom Metric respescts the base :meth:`ashpy.metrics.metric.Metric.__init__()`

.. warning::
    The ``name`` argument of the :meth:`ashpy.metrics.metric.Metric.__init__()` is a :obj:`str` identifier
    which should be unique across all the metrics used by your :class:`ashpy.trainers.Trainer`.

* Each custom Metric should have a :meth:`ashpy.metrics.metric.Metric.update_state()` respecting the signature.

    * This method is invoked during the training and receives an :class:`ashpy.contexts.context.Context`
    * In this example, since we are working under the :class:`ashpy.trainers.classifier.ClassifierTrainer` we are using an :class:`ashpy.contexts.classifier.ClassifierContext`.
      for more information on the Context family of objects see AshPy Internals. TODO: Inernal Link

.. _TensorFlow Addons: https://www.tensorflow.org/addons/overview
