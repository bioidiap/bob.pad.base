.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:26:09 +01

.. _bob.pad.base.vanilla_pad_features:

======================
 Vanilla PAD features
======================

.. todo::

   Introduce vanilla-pad features:

      - db: filelist and interface class
      - checkpoints
      - dask

   Look at bob.bio.base vanilla-biometrics features
   Import from filedb_guide and high_level_db_interface_guide

Most of the available features are equivalent to the ones defined in :ref:`bob.bio.base.vanilla_biometrics_advanced_features`.
However, there are some variations, and those are presented below.

Database interface
==================

The database interface definition follows closely the one in :ref:`bob.bio.base.database_interface`. However, the naming of the methods to retrieve data is different:

- :py:meth:`database.fit_samples` returns the samples used to train the classifier;
- :py:meth:`database.predict_samples` returns the samples that will be used for evaluating the system.
