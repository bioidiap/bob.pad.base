.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:26:09 +01

.. _bob.pad.base.vanilla_pad_features:

======================
 Vanilla PAD features
======================

Most of the available features are equivalent to the ones defined in :any:`pipeline_simple_features`.
However, there are some variations, and those are presented below.

Database interface
==================

The database interface definition follows closely the one in :ref:`bob.bio.base.database_interface`. However, the naming of the methods to retrieve data is different:

- :py:meth:`database.fit_samples` returns the samples (or delayed samples) used to train the classifier;
- :py:meth:`database.predict_samples` returns the samples that will be used for evaluating the system. This is where the group (`dev` or `eval`) is specified.

A difference with the bob.bio.base database interface is the presence of an ``attack_type`` annotation. It stores the type of PAI to allow the scoring of each different type of attack separately.


File list interface
-------------------

A class with those methods returning the corresponding data can be implemented for each dataset, but an easier way to do it is with the `file list` interface.
This allows the creation of multiple protocols and various groups by editing some CSV files.

The dataset configuration file will then be as simple as:

.. code-block:: python

   from bob.pad.base.database import CSVPADDataset

   database = CSVPADDataset("path/to/my_dataset", "my_protocol")

And the command to run an experiment with that configuration on the `svm-frames` pipeline::

$ bob pad vanilla-pad -d my_db_config_file.py svm-frames -o output_dir


The files must follow the following structure and naming:

.. code-block:: text

  my_dataset
  |
  +-- my_protocol
      |
      +-- train
      |   |
      |   +-- for_real.csv
      |   +-- for_attack.csv
      |
      +-- dev
      |   |
      |   +-- for_real.csv
      |   +-- for_attack.csv
      |
      +-- eval
          |
          +-- for_real.csv
          +-- for_attack.csv

The content of the files in the ``train`` folder is used when a protocol contains data for training the classifier.
The files in the ``eval`` folder are optional and are used in case a protocol contains data for evaluation.

These CSV files should contain at least the path to raw data and an identifier to the identity of the subject in the image (subject field).
The structure of each CSV file should be as below:

.. code-block:: text

   PATH,SUBJECT
   path_1,subject_1
   path_2,subject_2
   path_i,subject_j
   ...


Metadata can be shipped within the Samples (e.g gender, age, annotations, ...) by adding a column in the CSV file for each metadata:

.. code-block:: text

   PATH,SUBJECT,TYPE_OF_ATTACK,GENDER,AGE
   path_1,subject_1,A,B,C
   path_2,subject_2,A,B,1
   path_i,subject_j,2,3,4
   ...


Checkpoints and Dask
====================

In the same way as in :any:`bob.bio.base <pipeline_simple_features>`, it is possible to activate the checkpointing of experiments by passing the ``-c`` (``--checkpoint``) option in the command line.

The Dask integration can also be used by giving a client configuration to the ``-l`` (``--dask-client``) argument.
Basic Idiap SGE configurations are defined by bob.pipelines: ``sge`` and ``sge-gpu``::

$ bob pad vanilla-pad replay-attack svm-frames -o output_dir -l sge -c
