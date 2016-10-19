.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

.. _bob.pad.base.installation:

=========================
Installation Instructions
=========================

As noted before, this package is part of the ``bob.pad`` packages, which in turn are part of the signal-processing and machine learning toolbox Bob_.
To install `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_, please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

.. note::
  Currently, running Bob_ under MS Windows in not supported.
  However, we found that running Bob_ in a virtual Unix environment such as the one provided by VirtualBox_ is a good alternative.

The most simple and most convenient way to use the ``bob.pad`` tools is to use a ``zc.buildout`` package, as explained in more detail `here <https://github.com/idiap/bob/wiki/Installation#using-zcbuildout-for-production>`__.
There, in the ``eggs`` section of the ``buildout.cfg`` file, simply list the ``bob.pad`` packages that you want, like:

.. code-block:: python

   eggs = bob.pad.base
          bob.pad.voice
          bob.db.avspoof
          gridtk

in order to download and install all packages that are required for your experiments.
Running the simple command line:

.. code-block:: sh

   $ python bootstrap-buildout.py
   $ ./bin/buildout

will the download and install all dependent packages locally (relative to your current working directory), and create a ``./bin`` directory containing all the necessary scripts to run the experiments.


Databases
~~~~~~~~~

With ``bob.pad`` you will run biometric recognition experiments using databases that contain presentation attacks.
Though the PAD protocols are implemented in ``bob.pad``, the original data are **not included**.
To download the original data of the databases, please refer to the according Web-pages.
For a list of supported databases including their download URLs, please refer to the `spoofing_databases <https://gitlab.idiap.ch/bob/bob/wikis/Packages>`_.

After downloading the original data for the databases, you will need to tell ``bob.pad``, where these databases can be found.
For this purpose, we have decided to implement a special file, where you can set your directories.
Similar to ``bob.bio.base``, by default, this file is located in ``~/.bob_bio_databases.txt``, and it contains several lines, each line looking somewhat like:

.. code-block:: text

   [DEFAULT_DATABASE_DIRECTORY] = /path/to/your/directory

.. note::
   If this file does not exist, feel free to create and populate it yourself.


Please use ``./bin/databases.py`` for a list of known databases, where you can see the raw ``[YOUR_DATABASE_PATH]`` entries for all databases that you haven't updated, and the corrected paths for those you have.


.. note::
   If you have installed only ``bob.pad.base``, there is no database listed -- as all databases are included in other extension packages, such as ``bob.pad.voice``.


Test your Installation
~~~~~~~~~~~~~~~~~~~~~~

One of the scripts that were generated during the bootstrap/buildout step is a test script.
To verify your installation, you should run the script running the nose tests for each of the ``bob.pad`` packages:

.. code-block:: sh

  $ ./bin/nosetests -vs bob.pad.base
  $ ./bin/nosetests -vs bob.pad.voice
  ...

In case any of the tests fail for unexplainable reasons, please file a bug report through the `GitHub bug reporting system`_.

.. note::
  Usually, all tests should pass with the latest stable versions of the Bob_ packages.
  In other versions, some of the tests may fail.


Generate this documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, the documentation of this package is `available online <http://pythonhosted.org/bob.pad.base>`__, and this should be your preferred resource.
However, to generate this documentation locally, you call:

.. code-block:: sh

  $ ./bin/sphinx-build doc sphinx

Afterward, the documentation is available and you can read it, e.g., by using:

.. code-block:: sh

  $ firefox sphinx/index.html


.. _buildout.cfg: file:../buildout.cfg

.. include:: links.rst
