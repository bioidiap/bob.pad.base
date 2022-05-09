.. _bob.pad.base.implemented:

===================================
 Tools implemented in bob.pad.base
===================================

Please not that some parts of the code in this package are dependent on and reused from :ref:`bob.bio.base <bob.bio.base>` package.

Summary
-------

Base Classes
~~~~~~~~~~~~

Most of the base classes are reused from :ref:`bob.bio.base <bob.bio.base>`.
Only one base class that is presentation attack detection specific, ``Algorithm`` is implemented in this package.

Implementations
~~~~~~~~~~~~~~~

.. autosummary::
   bob.pad.base.pipelines.vanilla_pad.Database
   bob.pad.base.database.FileListPadDatabase

Preprocessors and Extractors
----------------------------

Preprocessors and Extractors from the :ref:`bob.bio.base <bob.bio.base>`
package can also be used in this package.


Databases
---------

.. automodule:: bob.pad.base.database

Grid Configuration
------------------
Code related to grid is reused from :ref:`bob.bio.base <bob.bio.base>` package. Please see the corresponding documentation.


.. include:: links.rst
