.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

.. _bob.pad.base.installation:


====================
 Setup Instructions
====================

To install any Bob_ package, please read the `Installation Instructions
<bobinstall_>`_. You will also need to prepare your biometric databases before
you can run experiments.


Databases
---------

With ``bob.pad`` you will run biometric recognition experiments using databases
that contain presentation attacks. Though the PAD protocols are implemented in
``bob.pad``, the original data are **not included**. To download the original
data of the databases, please refer to their corresponding Web-pages.

After downloading the original data for the databases, you will need to tell
``bob.pad``, where these databases can be found. For this purpose, a command
exists to define your directories:

.. code-block:: sh

   $ bob config set bob.db.<dbname> /path/to/the/db/data/folder

where ``<dbname>`` is the name of the database. For more information on the
``bob config`` command, see :ref:`bob.extension.rc`.

.. include:: links.rst
