.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:26:02 +01

.. _bob.pad.base.vanilla_pad_intro:

========================================================================
 Vanilla PAD: Introduction to presentation attack detection in practice
========================================================================


To easily run experiments in PAD, we offer a generic command called ``bob pad pipelines``.
Such CLI command is an entry point to several pipelines, and this documentation will focus on the one called **vanilla-pad**.

The following will introduce how a simple experiment can be run with this tool, from the sample data to a set of metrics and plots, as defined in :ref:`bob.pad.base.pad_intro`.


Running a biometric experiment with vanilla-pad
===============================================

A PAD experiment consists of taking a set of biometric `bonafide` and `impostor` samples, feeding them to a pipeline, to finally gather the corresponding set of scores for analysis.

.. figure:: img/vanilla_pad_pipeline.png
   :figwidth: 75%
   :align: center
   :alt: Data is fed to the pipeline either for training (to fit) or for evaluation (to transform and predict).

   The pipeline of Transformer(s) and Classifier can be trained (fit) or used to generate a score for each input sample.

Similarly to ``pipeline simple`` in bob.bio packages, the ``vanilla-pad`` command needs a pipeline configuration argument to specify which experiment to run and a database argument to indicate what data will be used. These can be given with the ``-p`` (``--pipeline``) and ``-d`` (``--database``) options, respectively::

$ bob pad vanilla-pad [OPTIONS] -p <pipeline> -d <database>

The different available options can be listed by giving the ``--help`` flag to the command::

$ bob pad vanilla-pad --help


Pipeline
--------

The `pipeline` argument given to vanilla-pad can be either a pipeline `resource name`, or a filename pointing to a configuration file defining the ``pipeline`` variable.

A list of existing `resource names` can be listed with::

$ resources.py -t pipeline


Database
--------

Similarly to `pipeline`, the `database` argument can be in the form of a predefined `resource name`, or a filename pointing to a file defining the ``database`` variable.

The list of database `resource names` can be retrieved with::

$ resources.py -t database


Building your own Vanilla PAD pipeline
======================================

The Vanilla PAD pipeline is the backbone of any experiment in this library. It is composed of:

   - Transformers: One or multiple instances in series of :py:class:`sklearn.base.BaseEstimator` and :py:class:`sklearn.base.TransformerMixin`. A Transformer takes a sample as input applies a modification on it and outputs the resulting sample. A transformer can be trained before being used.

   - A classifier: A class implementing the :py:meth:`fit` and :py:meth:`predict` methods. A Classifier takes a sample as input and returns a score. It is possible to train it beforehand with the :py:meth:`fit` method.


Transformers
------------

A Transformer is a class that implements the fit and transform methods, which allow the application of an operation on a sample of data.
For more details, see :ref:`bob.bio.base.transformer`.

Here is a basic stateless Transformer class:

.. code-block:: python

   from sklearn.base import TransformerMixin, BaseEstimator

   class MyTransformer(TransformerMixin, BaseEstimator):

      def fit(self, X, y):
         return self

      def transform(self, X):
         return modify_sample(X)


Classifier
----------

A Classifier is the final process of a Vanilla PAD pipeline.
Its goal is to decide if a transformed sample given as input is originating from a genuine sample or if an impostor is trying to be recognized as someone else.
The output is a score for each input sample.

Here is the minimal structure of a classifier:

.. code-block:: python

   class MyClassifier():
      def __init__(self):
         self.state = 0

      def fit(self, X, y):
         self.state = update_state(self.state, X, y)

      def predict(self, X):
         return do_prediction(self.state, X)

      def decision_function(self, X):
         return do_decision(X)

.. note::

   The easiest method is to use a scikit-learn classifier, like :py:class:`sklearn.svm.SVC`.
   They are compatible with our pipelines, on the condition to wrap them correctly (see :ref:`below <bob.pad.base.using_sklearn_classifiers>`).


Running an experiment
=====================

Two parts of an experiment have to be executed:

- **Fit**: labeled data is fed to the system to train the algorithm to recognize attacks and licit proprieties.
- **Predict**: assessing a series of test samples for authenticity, generating a score for each one.

These steps are chained together in a pipeline object used by the ``vanilla-pad`` command.
To build such a pipeline, the following configuration file can be created:

.. code-block:: python

   from sklearn.pipeline import Pipeline

   my_transformer = MyTransformer()

   my_classifier = MyClassifier()

   pipeline = Pipeline(
      [
         ("my_transformer", my_transformer),
         ("classifier", my_classifier),
      ]
   )

The pipeline can then be executed with the command::

$ bob pad vanilla-pad -d my_database_config.py -p my_pipeline_config.py -o output_dir

When executed with vanilla-pad, every training sample will pass through the pipeline, executing the ``fit`` methods.
Then, every sample of the `dev` set (and/or the `eval` set) will be given to the `transform` method of ``my_transformer`` and the result is passed to the ``decision_function`` method of ``my_classifier``.
The output of the classifier (scores) is written to a file.

.. note::

   By default, vanilla-pad expects the classifier to have a `decision_function` method to call for the prediction step. It can be changed with the '-f' switch to the prediction method of your classifier, for instance `-f predict_proba` to use this method of your scikit-learn classifiers.
   The usual `decision_function` of scikit-learn is their `predict_proba` method.


.. _bob.pad.base.using_sklearn_classifiers:

Using scikit-learn classifiers
------------------------------

To use an existing scikit-learn Transformer or Classifier, they need to be wrapped with a `SampleWrapper` (using :py:meth:`bob.pipelines.wrap`) to handle our :py:class:`~bob.pipelines.Sample` objects:

.. code-block:: python

   import bob.pipelines
   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC

   my_transformer = MyTransformer()


   sklearn_classifier = SVC()
   wrapped_classifier = bob.pipelines.wrap(
      ["sample"], sklearn_classifier, fit_extra_arguments=[("y", "is_bonafide")],
   )

   pipeline = Pipeline(
      [
         ("my_transformer", my_transformer),
         ("classifier", wrapped_classifier),
      ]
   )


Scores
------

Executing the vanilla-pad pipeline results in a list of scores, one for each
input sample compared against each registered model.
Depending on the chosen ScoreWriter, these scores can be in CSV (default), or 4 columns
lst file format (using the ``--csv-scores`` or ``--lst-scores`` options).
By default, the scores are written in the specified output directory (pointed to
vanilla-pad with the ``-o`` option), and in the CSV format, containing metadata in
additional columns (as opposed to the 4 columns format having no metadata).

The scores represent the performance of a system on that data, but are not easily
interpreted "as is", so evaluation scripts are available to analyze them and show
different aspects of the system performance.

.. figure:: img/vanilla_pad_pipeline_with_eval.png
   :figwidth: 75%
   :align: center
   :alt: The data is fed to the vanilla-pad pipeline, which produces scores files. Scripts allow the evaluation with metrics and plots.

   The vanilla-pad pipeline generates score files that can be used with various scripts to evaluate the system performance by computing metrics or drawing plots.


Evaluation
----------

Once the scores are generated for each class and group, the evaluation tools can be used to assess the performance of the system, by either drawing plots or computing metrics values at specific operation points.

Generally, the operation thresholds are computed on a specific set (development set or `dev`). Then those threshold values are used to compute the system error rates on a separate set (evaluation set or `eval`).

To retrieve the most common metrics values for a spoofing scenario experiment, run the following command:

.. code-block:: none

   $ bob pad metrics -e scores-{dev,eval} --legends ExpA

   Threshold of 11.639561 selected with the bpcer20 criteria
   ======  ========================  ===================
   ExpA    Development scores-dev    Eval. scores-eval
   ======  ========================  ===================
   APCER   5.0%                      5.0%
   BPCER   100.0%                    100.0%
   ACER    52.5%                     52.5%
   ======  ========================  ===================

   Threshold of 3.969103 selected with the eer criteria
   ======  ========================  ===================
   ExpA    Development scores-dev    Eval. scores-eval
   ======  ========================  ===================
   APCER   100.0%                    100.0%
   BPCER   100.0%                    100.0%
   ACER    100.0%                    100.0%
   ======  ========================  ===================

   Threshold of -0.870550 selected with the min-hter criteria
   ======  ========================  ===================
   ExpA    Development scores-dev    Eval. scores-eval
   ======  ========================  ===================
   APCER   100.0%                    100.0%
   BPCER   19.5%                     19.5%
   ACER    59.7%                     59.7%
   ======  ========================  ===================

.. note::
    When evaluation scores are provided, the ``-e`` option (``--eval``) must be passed.
    See metrics --help for further options.


Plots
-----

Customizable plotting commands are available in the :py:mod:`bob.pad.base` module.
They take a list of development and/or evaluation files and generate a single PDF
file containing the plots.

Available plots for a spoofing scenario (command ``bob pad``) are:

*  ``hist`` (Bonafide and PA histograms along with threshold criterion)

*  ``epc`` (expected performance curve)

*  ``gen`` (Generate random scores)

*  ``roc`` (receiver operating characteristic)

*  ``det`` (detection error trade-off)

*  ``evaluate`` (Summarize all the above commands in one call)

Use the ``--help`` option on the above-cited commands to find-out about more
options.


For example, to generate an EPC curve from development and evaluation datasets:

.. code-block:: sh

    $ bob pad epc -e -o 'my_epc.pdf' scores-{dev,eval}.csv

where `my_epc.pdf` will contain EPC curves for all the experiments.
