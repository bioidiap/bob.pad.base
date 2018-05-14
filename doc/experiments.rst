.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

.. _bob.pad.base.experiments:


===================================================
 Running Presentation Attack Detection Experiments
===================================================

Now, you are almost ready to run presentation attack detection (PAD) experiment.

Structure of a PAD Experiment
-----------------------------

Each biometric recognition experiment that is run with ``bob.pad`` is divided into the following several steps:

1. Data preprocessing: Raw data is preprocessed, e.g., for speech, voice activity is detected.
2. Feature extraction: Features are extracted from the preprocessed data.
3. Feature projector training: Models of genuine data and attacks are learnt.
4. Feature projection: The extracted features are projected into corresponding subspaces.
5. Scoring: The spoofing scores for genuine data and attacks are computed.
6. Evaluation: The computed scores are evaluated and curves are plotted.

These 6 steps are divided into four distinct groups:

* Preprocessing (step 1)
* Feature extraction (step 2)
* Attack detection (steps 3 to 5)
* Evaluation (step 6)

The communication between two steps is file-based, usually using a binary HDF5_ interface, which is implemented in the :py:class:`bob.io.base.HDF5File` class.
The output of one step usually serves as the input of the subsequent step(s).
Depending on the algorithm, some of the steps are not applicable/available.
``bob.pad`` takes care that always the correct files are forwarded to the subsequent steps.


.. _running_part_1:

Running Experiments
-------------------

To run an experiment, we provide a generic script ``./bin/spoof.py``.
To get a complete list of command line options, please run:

.. code-block:: sh

   $ ./bin/spoof.py --help

.. note::
   Sometimes, command line options have a long version starting with ``--`` and a short one starting with a single ``-``.
   In this section, only the long names of the arguments are listed, please refer to ``./bin/spoof.py --help``.

There are five command line options, which are required and sufficient to define the complete biometric recognition experiment.
These five options are:

* ``--database``: The database to run the experiments on
* ``--preprocessor``: The data preprocessor
* ``--extractor``: The feature extractor
* ``--algorithm``: The presentation attack detection algorithm
* ``--sub-directory``: A descriptive name for your experiment, which will serve as a sub-directory

The first four parameters, i.e., the ``database``, the ``preprocessor``, the ``extractor`` and the ``algorithm`` can be specified in several different ways.
For the start, we will use only the registered :ref:`Resources <bob.bio.base.resources>`.
These resources define the source code that will be used to compute the experiments, as well as all the meta-parameters of the algorithms (which we will call the *configuration*).
To get a list of registered resources, please call:

.. code-block:: sh

   $ ./bin/resources.py

Each package in ``bob.pad`` defines its own resources, and the printed list of registered resources differs according to the installed packages.
If only ``bob.pad.base`` is installed, no databases and no preprocessors will be listed.

.. note::
   You will also find some ``grid`` resources being listed.
   These type of resources will be explained :ref:`later <running_in_parallel>`.

One command line option, which is not required, but recommended, is the ``--verbose`` option.
By default, the algorithms are set up to execute quietly, and only errors are reported.
To change this behavior, you can use the ``--verbose`` option several times to increase the verbosity level to show:

1) Warning messages
2) Informative messages
3) Debug messages

When running experiments, it is a good idea to set verbose level 2, which can be enabled by using the short version: ``-vv``.
So, a typical PAD experiment (in this case, attacks detection in speech) would look like the following:

.. code-block:: sh

   $ ./bin/spoof.py --database <database-name> --preprocessor <preprocessor> --extractor <extractor> --algorithm <algorithm> --sub-directory <folder_name> -vv

Before running an experiment, it is recommended to add the ``--dry-run`` option, so that it will only print, which steps would be executed, without actually executing them, and make sure that everything works as expected.

The final result of the experiment will be one (or more) score file(s).
Usually, they will be called something like ``scores-dev-real`` for genuine data, ``scores-dev-attack`` for attacks, and ``scores-dev`` for the results combined in one file.
By default, you can find them in a sub-directory the ``result`` directory, but you can change this option using the ``--result-directory`` command line option.

.. note::
   At Idiap_, the default result directory differs, see ``./bin/spoof.py --help`` for your directory.


.. _bob.pad.base.evaluate:

Evaluating Experiments
----------------------

After the experiment has finished successfully, one or more text file containing
all the scores are written. In this section, commands that helps to quickly
evaluate a set of scores by generating metrics or plots are presented here.
The scripts take as input either a 4-column or 5-column data format as specified
in the documentation of :py:func:`bob.bio.base.score.load.four_column` or 
:py:func:`bob.bio.base.score.load.five_column`.

Metrics
=======

Several metrics based on a selected thresholds (bpcer20: when APCER is set to 5%,
eer, when BPCER == APCER and min-hter, when HTER is minimum) on the development
set and apply them on evaluation sets (if provided) are generated used
``metrics`` command. The reported `standard metrics`_ are:

*   APCER: Attack Presentation Classification Error Rate

*   BPCER: Bona-fide Presentation Classification Error Rate

*   HTER (non-ISO): Half Total Error Rate ((BPCER+APCER)/2)

For example:

.. code-block:: sh

    $ bob pad metrics scores-{dev,test} --titles ExpA

    Threshold of 6.624767 selected with the bpcer20 criteria
    ========  ========================  ===================
    ExpA      Development scores-dev    Eval. scores-eval
    ========  ========================  ===================
    BPCER20   5.0%                      5.0%
    EER       0.0%                      0.0%
    min-HTER  2.5%                      2.5%
    ========  ========================  ===================

    Threshold of 6.534215 selected with the eer criteria
    ========  ========================  ===================
    ExpA      Development scores-dev    Eval. scores-eval
    ========  ========================  ===================
    BPCER20   6.1%                      6.1%
    EER       0.0%                      0.0%
    min-HTER  3.0%                      3.0%
    ========  ========================  ===================

    Threshold of 6.534215 selected with the min-hter criteria
    ========  ========================  ===================
    ExpA      Development scores-dev    Eval. scores-eval
    ========  ========================  ===================
    BPCER20   6.1%                      6.1%
    EER       0.0%                      0.0%
    min-HTER  3.0%                      3.0%
    ========  ========================  ===================

.. note::
    You can compute analysis on development set(s) only by passing option
    ``--no-evaluation``. See metrics --help for further options.

Plots
=====

Customizable plotting commands are available in the :py:mod:`bob.pad.base` module.
They take a list of development and/or evaluation files and generate a single PDF
file containing the plots. Available plots are:

*  ``hist`` (Bona fida and PA histograms along with threshold criterion)

*  ``vuln`` (Vulnerability analysis distributions)

*  ``epc`` (expected performance curve)

*  ``epsc`` (expected performance spoofing curve)

Use the ``--help`` option on the above-cited commands to find-out about more
options.


For example, to generate a EPSC curve from development and evaluation datasets:

.. code-block:: sh

    $bob pad epsc -o 'my_epsc.pdf' scores-{dev,test}

where `my_epsc.pdf` will contain EPSC curves for all the experiments.

.. note::
    IAPMR curve can be plotted along with EPC and EPSC using option
    ``--iapmr``. 3D EPSC can be generated using the ``--three-d``. See metrics
    --help for further options.

.. _running_in_parallel:

Running in Parallel
-------------------

One important property of the ``./bin/spoof.py`` script is that it can run in parallel, using either several threads on the local machine, or an SGE grid.
To achieve that, ``bob.pad`` is well-integrated with our SGE grid toolkit GridTK_, which we have selected as a python package in the :ref:`Installation <bob.pad.base.installation>` section.
The ``./bin/spoof.py`` script can submit jobs either to the SGE grid, or to a local scheduler, keeping track of dependencies between the jobs.

The GridTK_ keeps a list of jobs in a local database, which by default is called ``submitted.sql3``, but which can be overwritten with the ``--gridtk-database-file`` option.
Please refer to the `GridTK documentation <http://pythonhosted.org/gridtk>`_ for more details on how to use the Job Manager ``./bin/jman``.

Two different types of ``grid`` resources are defined, which can be used with the ``--grid`` command line option.
The first type of resources will submit jobs to an SGE grid.
They are mainly designed to run in the Idiap_ SGE grid and might need some adaptations to run on your grid.
The second type of resources will submit jobs to a local queue, which needs to be run by hand (e.g., using ``./bin/jman --local run-scheduler --parallel 4``), or by using the command line option ``--run-local-scheduler``.
The difference between the two types of resources is that the local submission usually starts with ``local-``, while the SGE resource does not.

Hence, to run the same experiment as above using four parallel threads on the local machine, re-nicing the jobs to level 10, simply call:

.. code-block:: sh

   $ ./bin/spoof.py --database <database-name> --preprocessor <preprocessor> --extractor <extractor> --algorithm <algorithm> --sub-directory <folder_name> -vv --grid local-p4 --run-local-scheduler --nice 10

.. note::
   You might realize that the second execution of the same experiment is much faster than the first one.
   This is due to the fact that those parts of the experiment, which have been successfully executed before (i.e., the according files already exist), are skipped.
   To override this behavior, i.e., to always regenerate all parts of the experiments, you can use the ``--force`` option.


Command Line Options to change Default Behavior
===============================================
Additionally to the required command line arguments discussed above, there are several options to modify the behavior of the experiments.
One set of command line options change the directory structure of the output.
By default, intermediate (temporary) files are by default written to the ``temp`` directory, which can be overridden by the ``--temp-directory`` command line option, which expects relative or absolute paths:

Re-using Parts of Experiments
=============================
If you want to re-use parts previous experiments, you can specify the directories (which are relative to the ``--temp-directory``, but you can also specify absolute paths):

* ``--preprocessed-data-directory``
* ``--extracted-directory``
* ``--projected-directory``

or even trained projector, i.e., the results of the projector:

* ``--projector-file``

For that purpose, it is also useful to skip parts of the tool chain.
To do that you can use:

* ``--skip-preprocessing``
* ``--skip-extraction``
* ``--skip-projector-training``
* ``--skip-projection``
* ``--skip-score-computation``

although by default files that already exist are not re-created.
You can use the ``--force`` argument combined with the ``--skip...`` arguments (in which case the skip is preferred).
To run just a sub-selection of the tool chain, you can also use the ``--execute-only`` option, which takes a list of options out of: ``preprocessing``, ``extraction``, ``projector-training``, ``projection``, or ``score-computation``.


Database-dependent Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Many databases define several protocols that can be executed.
To change the protocol, you can either modify the configuration file, or simply use the ``--protocol`` option.

Some databases define several kinds of evaluation setups.
For example, often two groups of data are defined, a so-called *development set* and an *evaluation set*.
The scores of the two groups will be concatenated into several files called **scores-dev** and **scores-eval**, which are located in the score directory (see above).
In this case, by default only the development set is employed.
To use both groups, just specify ``--groups dev eval`` (of course, you can also only use the ``'eval'`` set by calling ``--groups eval``).

.. include:: links.rst
.. _`standard metrics`: https://www.iso.org/obp/ui/#iso:std:iso-iec:30107:-3:ed-1:v1:en
