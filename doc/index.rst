.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

.. _bob.pad.base:

===================================================
 Running Presentation Attack Detection Experiments
===================================================

The ``bob.pad`` packages provide open source tools to run comparable and
reproducible presentation attack detection (PAD) experiments. The API of running
experiments is designed on top of sklearn_'s API to be as simple and familiar as
possible. On top of that, we use :ref:`bob.pipelines`'s sample-based API to
extend the sklearn_'s API to work with sample-based data. Make sure you have
reviewed the documentation of :ref:`bob.pipelines` before continuing here.

This package is the base package of the PAD framework. It provides the core
functionality to run PAD experiments and to evaluate the results. However, the
implementation of the PAD algorithms are provided in modality specific packages
such as:

* `bob.pad.face <http://pypi.python.org/pypi/bob.pad.face>`_
* `bob.pad.voice <http://pypi.python.org/pypi/bob.pad.voice>`_ (at the moment of
  writing, this package is not maintained anymore and does not work with the new
  version of ``bob.pad.base``.)

Follow the documentation below to learn how to use the PAD framework.


Users Guide
=============

.. toctree::
    :maxdepth: 2

    setup
    generic_intro
    intro
    features


Reference Manual
==================

.. toctree::
    :maxdepth: 2

    py_api


Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
