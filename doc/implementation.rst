.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. author: Pavel Korshunov <pavel.korshunov@idiap.ch>
.. date: Wed Apr 27 14:58:21 CEST 2016

======================
Implementation Details
======================

The ``bob.pad`` set of modules are specifically designed to be as flexible as possible while trying to keep things simple.
Therefore, python is used to implement tools such as preprocessors, feature extractors and the PAD algorithms.
Everything is file based so any tool can implement its own way of reading and writing data, features or models.
Configurations are stored in configuration files, so it should be easy to test different parameters of your algorithms without modifying the code.

Also, it is important to note that ``bob.pad.base`` reuses concept, design, and large parts of code from :ref:`bob.bio.base <bob.bio.base>` package,
which is designed for recognition experiments.


Base Classes
------------

Most of the functionality is provided in the base classes, but any function can be overridden in the derived class implementations.

In the derived class constructors, the base class constructor needs to be called.
For automatically tracing the algorithms, all parameters that are passed to the derived class constructor should be passed to the base class constructor as a list of keyword arguments (which is indicated by ``...`` below).
This will assure that all parameters of the experiments are stored into the ``Experiment.info`` file.

.. note::
   All tools are based on reading, processing and writing files.
   By default, any type of file is allowed to be handled, and file names are provided to the ``read_...`` and ``write_...`` functions as strings.

If you plan to write your own tools, please assure that you are following the following structure.


.. _bob.bio.base.preprocessors:
.. _bob.bio.base.extractors:

Preprocessors and Extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All preprocessor and extractor classes are based on the empty base classess implemented in :ref:`bob.bio.base <bob.bio.base>`, specifically,
on :py:class:`bob.bio.base.preprocessor.Preprocessor` and :py:class:`bob.bio.base.extractor.Extractor` classes.

.. _bob.pad.base.algorithms:

Algorithms
~~~~~~~~~~
All presentation attack detection  algorithms are derived from the :py:class:`bob.pad.base.algorithm.Algorithm` class.
The constructor of this class has the following options, which are selected according to the current algorithm:

* ``performs_projection``: If set to ``True``, features will be projected using the ``project`` function.
  With the default ``False``, the ``project`` function will not be called at all.
* ``requires_projector_training``: If ``performs_projection`` is enabled, this flag specifies if the projector needs training.
  If ``True`` (the default), the ``train_projector`` function will be called.

A presentation attack detection algorithm has to override at least two functions:

* ``__init__(self, <parameters>)``: Initializes the face recognition algorithm with the parameters it needs.
  Calls the base class constructor, e.g. as ``bob.pad.base.algorithm.Algorithm.__init__(self, ...)`` (there are more parameters to this constructor, see above).
* ``score(self, toscore) -> value``: Computes score given the projected value returned by the classifier.

Additionally, an algorithm may need to project the features before they can be used in spoofing detection.
In this case, (some of) the function(s) are overridden:

* ``train_projector(self, train_features, projector_file)``: Uses the given list of features and writes the ``projector_file``.

  .. warning::
     If you write this function, please assure that you use both ``performs_projection=True`` and ``requires_projector_training=True`` (for the latter, this is the default, but not for the former) during the base class constructor call in your ``__init__`` function.
     Please also assure that you overload the ``project`` function.

* ``load_projector(self, projector_file)``: Loads the projector from the given file, i.e., as stored by ``train_projector``.
  This function is always called before the ``project`` and ``score`` functions are executed.
* ``project(self, feature) -> feature``: Projects the given feature and returns the projected feature, which should either be a :py:class:`numpy.ndarray` or an instance of a class that defines a ``save(bob.io.base.HDF5File)`` method.

  .. note::
     If you write this function, please assure that you use ``performs_projection=True`` during the base class constructor call in your ``__init__`` function.

And once more, if the projected feature is not of type ``numpy.ndarray``, the following methods are overridden:

* ``write_feature(feature, feature_file)``: Writes the feature (as returned by the ``project`` function) to file.
* ``read_feature(feature_file) -> feature``: Reads and returns the feature (as written by the ``write_feature`` function).


By default, it is assumed that features are of type :py:class:`numpy.ndarray`.
Finally, the :py:class:`bob.pad.base.algorithm.Algorithm` class provides default implementation for the case that several scores (or features) are used for one sample:

* ``score_for_multiple_projections(self, toscore)``: In case your object store several features or scores, **call** this function to compute the average (or min, max, ...) of the scores.


Implemented Tools
-----------------

Example implementations of the base classes can be found in all of the ``bob.pad`` packages.
Here is the current list of implementations:

* :ref:`bob.pad.voice <bob.pad.voice>` : :ref:`bob.pad.voice`


.. todo:: complete this list, once the other packages are documented as well.


Databases
---------

Databases provide information about the data sets, on which the PAD algorithm should run on.
Particularly, databases come with one or more evaluation protocols, which defines, which part of the
data should be used for training, algorithm tuning, and testing.
Some protocols split up the data into three different groups: a training set (aka ``train`` group), a
development set (aka ``dev`` group) and an evaluation set (aka ``eval``, sometimes also referred as test set).

Anti-spoofing Database Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most of the data sets, we rely on the database interfaces from Bob_.
Particularly, all databases that are derived from the :py:class:`bob.pad.base.database.PadDatabase` (click `here <https://gitlab.idiap.ch/bob/bob/wikis/Packages>`_ for a list of implemented databases) are supported.

Defining your own Database
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to define you own database, you are welcome to write your own database wrapper class.
In this case, you have to derive your class from the :py:class:`bob.pad.base.database.PadDatabase`,
and provide only the following functions:

* ``__init__(self, <your-parameters>, **kwargs)``: Constructor of your database interface.
  Please call the base class constructor, providing all the required parameters, e.g. by ``bob.pad.base.database.PadDatabase.__init__(self, **kwargs)``.
* ``objects(self, , groups=None, protocol=None, purposes=None, model_ids=None, **kwargs)``: Expected to return a list of :py:class:`bob.pad.base.database.PadFile` objects of the database given the specified parameters.
  The list needs to be sorted by the file id (you can use the ``self.sort(files)`` function for sorting).
* ``training_files(self, step, arrange_by_client = False)``: A sorted list of the :py:class:`bob.bio.base.database.BioFile` objects that is used for training.
  You should have ``arrange_by_clients`` disabled.


.. _bob.pad.base.configuration-files:

Configuration Files
-------------------

One important aspect of the ``bob.pad`` packages is reproducibility.
To be able to reproduce an experiment, it is required that all parameters of all tools are present.

In ``bob.pad`` (similarly to ``bob.bio``) this is achieved by providing these parameters in configuration files.
In these files, an *instance* of one of the tools is generated, and assigned to a variable with a specific name.
These variable names are:

* ``database`` for an instance of a (derivation of a) :py:class:`bob.pad.base.database.PadDatabase`
* ``preprocessor`` for an instance of a (derivation of a) :py:class:`bob.bio.base.preprocessor.Preprocessor`
* ``extractor`` for an instance of a (derivation of a) :py:class:`bob.bio.base.extractor.Extractor`
* ``algorithm`` for an instance of a (derivation of a) :py:class:`bob.pad.base.algorithm.Algorithm`
* ``grid`` for an instance of the :py:class:`bob.bio.base.grid.Grid`


.. _bob.bio.base.resources:

Resources
---------

Finally, some of the configuration files, which sit in the ``bob/pad/*/config`` directories, are registered as *resources*.
This means that a resource is nothing else than a short name for a registered instance of one of the tools (database, preprocessor, extractor, algorithm or grid configuration) of ``bob.pad``, which has a pre-defined set of parameters.

The process of registering a resource is relatively easy.
We use the SetupTools_ mechanism of registering so-called entry points in the ``setup.py`` file of the according ``bob.pad`` package.
Particularly, we use a specific list of entry points, which are:

* ``bob.pad.base.database.PadDatabase`` to register an instance of a (derivation of a) :py:class:`bob.pad.base.database.PadDatabase`
* ``bob.bio.preprocessor`` to register an instance of a (derivation of a) :py:class:`bob.bio.base.preprocessor.Preprocessor`
* ``bob.bio.extractor`` to register an instance of a (derivation of a) :py:class:`bob.bio.base.extractor.Extractor`
* ``bob.pad.algorithm`` to register an instance of a (derivation of a) :py:class:`bob.pad.base.algorithm.Algorithm`
* ``bob.bio.grid`` to register an instance of the :py:class:`bob.bio.base.grid.Grid`

For each of the tools, several resources are defined, which you can list with the ``./bin/resources.py`` command line.

When you want to register your own resource, make sure that your configuration file is importable (usually it is sufficient to have an empty ``__init__.py`` file in the same directory as your configuration file).
Then, you can simply add a line inside the according ``entry_points`` section of the ``setup.py`` file (you might need to create that section, just follow the example of the ``setup.py`` file that you can find online in the base directory of our `bob.pad.base GitHub page <http://github.com/bioidiap/bob.pad.base>`__).

After re-running ``./bin/buildout``, your new resource should be listed in the output of ``./bin/resources.py``.


.. include:: links.rst
