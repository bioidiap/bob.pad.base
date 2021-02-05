.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

.. _bob.pad.base.installation:


.. todo::

   update and link to general bob install instructions.


===========================
 Installation Instructions
===========================

As noted before, this package is part of the ``bob.pad`` packages, which in
turn are part of the signal processing and machine learning toolbox Bob_. To
install Bob_, please read the `Installation Instructions <bobinstall_>`_.

Then, to install the ``bob.pad`` packages and in turn, maybe the database
packages that you want to use, use conda_ to install them:

.. code-block:: sh

    $ conda search bob.pad # searching
    $ conda search bob.db  # searching
    $ conda install bob.pad.base bob.pad.<padname> bob.db.<dbname>

where you would replace ``<padname>`` and ``<dbname>`` with the name of
packages that you want to use.

An example installation
-----------------------

For example, to run a speech presentation attack detection experiments,
you need preprocessor, extractor, classifier, and
a database. Preprocessors and extractors can be reused from ``bob.bio`` packages, while classifier is normally
provided in ``bob.pad``. Hence for speech PAD, you can take
the :py:class:`bob.bio.spear.preprocessor.Energy_2Gauss` and the
:py:class:`bob.bio.spear.extractor.Cepstral` feature extractor defined in
:ref:`bob.bio.spear <bob.bio.spear>`, and the
:py:class:`bob.pad.base.algorithm.OneClassGMM` algorithm defined in
:ref:`bob.pad.base <bob.pad.base>`, using voicePA database (contains speech presentation attacks)
interface defined in :ref:`bob.db.voicepa <bob.db.voicepa>`. Running the
command line below will install all the required packages:

.. code-block:: sh

    $ conda activate <bob_conda_environment>
    $ conda install bob.bio.base \
                    bob.bio.spear \
                    bob.pad.base \
                    bob.pad.voice \
                    bob.db.voicepa \
                    gridtk

Databases
---------

With ``bob.pad`` you will run biometric recognition experiments using databases that contain presentation attacks.
Though the PAD protocols are implemented in ``bob.pad``, the original data are **not included**.
To download the original data of the databases, please refer to the corresponding Web-pages.
For a list of supported databases including their download URLs,
please refer to the `spoofing_databases <https://gitlab.idiap.ch/bob/bob/wikis/Packages>`_.

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


.. include:: links.rst
