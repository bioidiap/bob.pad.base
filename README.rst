.. vim: set fileencoding=utf-8 :
.. Pavel Korshunov <pavel.korshunov@idiap.ch>
.. Wed 30 Sep 23:36:23 2015 CET

========================================
Scripts to run anti-spoofing experiments
========================================

This package is part of the ``bob.pad`` packages, which allow to run comparable and reproducible presentation attack detection (PAD) experiments on publicly available databases.

This package contains basic functionality to run PAD experiments.
It provides a generic ``./bin/spoof.py`` script that takes several parameters, including:

* A database and its evaluation protocol
* A data preprocessing algorithm
* A feature extraction algorithm
* A PAD algorithm

All these steps of the PAD system are given as configuration files.

In this base class implementation, only a core functionality is implemented. The specialized algorithms should be provided by other packages, which are usually in the ``bob.pad`` namespace, such as:

* `bob.pad.voice for speech-related PAD algorithms


Installation
------------
To install this package -- alone or together with other `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_ -- please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

.. _bob: https://www.idiap.ch/software/bob
