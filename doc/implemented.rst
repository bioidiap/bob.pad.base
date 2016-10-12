.. _bob.bio.base.implemented:

=================================
Tools implemented in bob.pad.base
=================================

Please not that some parts of the code in this package are dependent on and reused from :ref:`bob.bio.base <bob.bio.base>` package.

Summary
-------

Base Classes
~~~~~~~~~~~~

Most of the base classes are reused from :ref:`bob.bio.base <bob.bio.base>`.
Only one base class that is presentation attack detection specific, ``Algorithm`` is implemented in this package.

.. autosummary::
   bob.pad.base.algorithm.Algorithm

Implementations
~~~~~~~~~~~~~~~

.. autosummary::
   bob.pad.base.database.PadDatabase
   bob.pad.base.database.PadFile

Preprocessors
-------------
Preprocessor is the same as in :ref:`bob.bio.base <bob.bio.base>` package.

.. automodule:: bob.bio.base.preprocessor


Extractors
----------
Extractor is the same as in :ref:`bob.bio.base <bob.bio.base>` package.

.. automodule:: bob.bio.base.extractor


Algorithms
----------

.. automodule:: bob.pad.base.algorithm

Databases
---------

.. automodule:: bob.pad.base.database

Grid Configuration
------------------
Code related to grid is reused from :ref:`bob.bio.base <bob.bio.base>` package. Please see the corresponding documentation.


.. include:: links.rst
