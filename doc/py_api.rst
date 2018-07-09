.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

===========================
Python API for bob.pad.base
===========================

Generic functions
-----------------


Tools to run PAD experiments
----------------------------

Command line generation
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.pad.base.tools.command_line_parser
   bob.pad.base.tools.initialize
   bob.pad.base.tools.command_line
   bob.pad.base.tools.write_info
   bob.pad.base.tools.FileSelector

Algorithm
~~~~~~~~~

.. autosummary::
   bob.pad.base.tools.train_projector
   bob.pad.base.tools.project
   bob.pad.base.algorithm

Scoring
~~~~~~~

.. autosummary::
   bob.bio.base.tools.compute_scores

Details
-------

.. automodule:: bob.pad.base


.. automodule:: bob.pad.base.tools

   .. autoclass:: FileSelector


.. include:: links.rst
