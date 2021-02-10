.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

.. _bob.pad.base:

===================================================
 Running Presentation Attack Detection Experiments
===================================================

The ``bob.pad`` packages provide open source tools to run comparable and reproducible presentation attack detection (PAD) experiments.
To design such experiment, one has to choose:

* a databases containing the original data, and a protocol that defines how to use the data,
* a data preprocessing algorithm, i.e., face detection for face PAD experiments or voice activity detection for speaker PAD,
* the type of features to extract from the preprocessed data,
* the presentation attack detection algorithm to employ,
* the way to evaluate the results

For any of these parts, several different types are implemented in the ``bob.pad`` packages, and basically any
combination of the five parts can be executed.
For each type, several meta-parameters can be tested.
This results in a nearly infinite amount of possible experiments that can be run using the current setup.
But it is also possible to use your own database, preprocessor, feature extractor, or PAD algorithm and test this against the baseline algorithms implemented in our packages.

.. note::
    The ``bob.pad.*`` packages are derived from the `bob.bio.* <http://pypi.python.org/pypi/bob.bio.base>`__, packages that are designed for biometric recognition experiments.

This package :py:mod:`bob.pad.base` includes the basic definition of a PAD experiment, as well as a generic script, which can execute the full experiment in a single command line.
Changing the employed tools such as the database, protocol, preprocessor, feature extractor or a PAD algorithm is as simple as changing a command line parameter.

The implementation of (most of) the tools is separated into other packages in the ``bob.pad`` namespace.
All these packages can be easily combined.
Here is a growing list of derived packages:

* `bob.pad.voice <http://pypi.python.org/pypi/bob.pad.voice>`__ Tools to run presentation attack detection experiments for speech, including several Cepstral-based features and LBP-based feature extraction, OneClassGMM-based and logistic regression based algorithms, as well as plot and score fusion scripts.
* `bob.pad.face <http://pypi.python.org/pypi/bob.pad.face>`__ Tools to run presentation attack detection experiments for face, including face-related feature extraction, OneClassGMM, SVM, and logistic regression based algorithms, as well as plotting scripts.

If you are interested, please continue reading:


Users Guide
=============

.. toctree::
    :maxdepth: 2

    installation
    pad_intro
    vanilla_pad_intro
    vanilla_pad_features


Reference Manual
==================

.. toctree::
    :maxdepth: 2

    implemented
    py_api


Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
