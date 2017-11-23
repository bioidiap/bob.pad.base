.. vim: set fileencoding=utf-8 :
.. @author: Manuel Guenther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

====================================
 User's Guide for PAD File List API
====================================

The low-level Database Interface
--------------------------------

The :py:class:`bob.pad.base.database.FileListPadDatabase` complies with the standard PAD database as described in :ref:`bob.pad.base`.
All functions defined in that interface are properly instantiated, as soon as the user provides the required file lists.

Creating File Lists
-------------------

The initial step for using this package is to provide file lists specifying the ``'train'`` (training), ``'dev'`` (development) and ``'eval'`` (evaluation) sets to be used by the PAD algorithm.
The summarized complete structure of the list base directory (here denoted as ``basedir``) containing all the files should be like this::

  basedir -- train -- for_real.lst
         |       |-- for_attack.lst
         |
         |-- dev -- for_real.lst
         |      |-- for_attack.lst
         |
         |-- eval -- for_real.lst
                 |-- for_attack.lst


The file lists should contain the following information for PAD experiments to run properly:

* ``filename``: The name of the data file, **relative** to the common root of all data files, and **without** file name extension.
* ``client_id``: The name or ID of the subject the biometric traces of which are contained in the data file.
  These names are handled as :py:class:`str` objects, so ``001`` is different from ``1``.
* ``attack_type``: This is not contained in `for_real.lst` files, only in `for_attack.lst` files.
  The type of attack (:py:class:`str` object).


The following list files need to be created:

- **For real**:

  * *real file*, with default name ``for_real.lst``, in the default sub-directories ``train``, ``dev`` and ``eval``, respectively.
    It is a 2-column file with format:

    .. code-block:: text

      filename client_id

  * *attack file*, with default name ``for_attack.lst``, in the default sub-directories ``train``, ``dev`` and ``eval``, respectively.
    It is a 3-column file with format:

    .. code-block:: text

      filename client_id attack_type


.. note:: If the database does not provide an evaluation set, the ``eval`` files can be omitted.


Protocols and File Lists
------------------------

When you instantiate a database, you have to specify the base directory that contains the file lists.
If you have only a single protocol, you could specify the full path to the file lists described above as follows:

.. code-block:: python

  >>> db = bob.pad.base.database.FileListPadDatabase('basedir/protocol')

Next, you should query the data, WITHOUT specifying any protocol:

.. code-block:: python

  >>> db.objects()

Alternatively, if you have more protocols, you could do the following:

.. code-block:: python

  >>> db = bob.pad.base.database.FileListPadDatabase('basedir')
  >>> db.objects(protocol='protocol')

When a protocol is specified, it is appended to the base directory that contains the file lists.
This allows to use several protocols that are stored in the same base directory, without the need to instantiate a new database.
For instance, given two protocols 'P1' and 'P2' (with filelists contained in 'basedir/P1' and 'basedir/P2', respectively), the following would work:

.. code-block:: python

  >>> db = bob.pad.base.database.FileListPadDatabase('basedir')
  >>> db.objects(protocol='P1') # Get the objects for the protocol P1
  >>> db.objects(protocol='P2') # Get the objects for the protocol P2


The high-level Database Interface
---------------------------------

the low-level FileList database interface is extended, so that filelist databases can be used to run both types:
vulnerability analysis experiments using :ref:`bob.bio.base <bob.bio.base>` verification framework
and PAD experiments using ``bob.pad.base`` framework.

For instance, provided the lists of files for database ``example_db`` in the correct format are located
inside ``lists`` directory (i.e., inside ``lists/example_db``), the PAD and verification versions of the
database can be created as following:

.. code-block:: python

  >>> from bob.pad.base.database import HighBioDatabase, HighPadDatabase
  >>> pad_db = HighPadDatabase(db_name='example_db')
  >>> bio_db = HighBioDatabase(db_name='example_db')



