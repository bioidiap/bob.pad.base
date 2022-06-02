.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:26:09 +01

.. _bob.pad.base.features:

===================
 Advanced features
===================

There are several extra features that we did not discuss. In this section we'll
explain the database interface, checkpointing of experiments, and multitasking
with Dask.

Database interface
==================

All PAD databases must inherit from the :py:class:`bob.pad.base.pipelines.Database` class
and implement the following methods:

- ```database.fit_samples``` returns the samples (or delayed samples) used
  to train the classifier;
- ```database.predict_samples``` returns the samples that will be used for
  evaluating the system. This is where the group (`dev` or `eval`) is specified.

The returned samples must have the following attributes:

- ``data``: the data of the sample
- ``key``: a unique identifier for the sample. must be a string.
- ``attack_type``: the attack type of the sample, must be ``None`` for bonafide
  samples. This will indicate the presentation attack instrument (PAI) of the
  attack sample and will be used to report error rates per PAI.
- ``subject_id``: The identity of the subject. This might not be available for
  all databases.


File list interface
-------------------

A class with those methods returning the corresponding data can be implemented
for each dataset, but an easier way to do it is with the *file list* interface.
This allows the creation of multiple protocols and various groups by editing
some CSV files. The :py:class:`bob.pad.base.database.FileListPadDatabase` class,
which builds on :ref:`bob.pipelines.csv_database`, implements this interface.

The dataset configuration file will can be as simple as:

.. code-block:: python

   from bob.pad.base.database import FileListPadDatabase

   database = FileListPadDatabase("path/to/my_dataset", "my_protocol")

The files must follow the following structure and naming:

.. code-block:: text

  my_dataset
  |
  +-- my_protocol
      |
      +-- train.csv
      +-- dev.csv
      +-- eval.csv

The ``dev.csv`` file is the main file here and is used for scoring samples of
the development group. The content of the ``train.csv`` file is used when a
protocol contains data for training the classifier. The ``eval.csv`` file is
optional and is used in case a protocol contains data for evaluation.

These CSV files should contain at least the path to raw data and an identifier
to the identity of the subject in the image (subject field) and an attack type.
The structure of each CSV file should be as below:

.. code-block:: text

   filename,subject,attack_type
   path_1,subject_1,
   path_2,subject_2,
   path_3,subject_1,attack_1
   path_4,subject_2,attack_1
   ...

The ``attack_type`` field is used to differentiate bonafide presentations from
attacks. An empty field indicates a bonafide sample. Otherwise different attack
types can be used (e.g. ``print``, ``replay``, etc.), and can be analyzed
separately during evaluation.

Metadata can be shipped within the Samples (e.g gender, age, session, ...) by
adding a column in the CSV file for each metadata:

.. code-block:: text

   filename,subject,attack_type,gender,age
   path_1,subject_1,,M,25
   path_2,subject_2,,F,24
   paht_3,subject_1,attack_1,M,25
   paht_4,subject_2,attack_1,F,24
   ...


Checkpoints and Dask
====================

By default, the ``bob pad run-pipeline`` command will save the features of each step of the pipeline
and the fitted estimators in the output folder. To avoid this, use the ``--memory`` option.

The Dask integration can also be used by giving a client configuration to the
``--dask-client`` option. Basic Idiap SGE configurations are defined by
bob.pipelines: ``sge`` and ``sge-gpu``::

   $ bob pad run-pipeline --output output_dir --dask-client sge ...

.. note::

   You may want to read the Dask section in
   :ref:`bob.bio.base.pipeline_simple_advanced_features` as well for more
   in-depth information.
