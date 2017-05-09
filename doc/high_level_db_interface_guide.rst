.. vim: set fileencoding=utf-8 :
.. @author: Olegs Nikisins <olegs.nikisins@idiap.ch>
.. @date:   May 2017


=============================================
High Level Database Interface How-To Guide
=============================================

The *high level database interface* (HLDI) is needed to run biometric experiments using non-filelist databases (e.g. if one wants to use SQL-based database package).

This tutorial explains how to create a *high level* database
interface, using as an example ``bob.pad.*`` framework (e.g.
``bob.pad.face``). The process is similar for ``bob.bio`` frameworks,
e.g. ``bob.bio.face``, ``bob.bio.vein``). High level database interface
is a link between low level database interface/package (e.g. ``bob.db.replay``) and a
corresponding framework used to run biometric experiments (e.g.
``bob.pad.face``). Generally speaking, the low level interface has lot's
of querying options, which are not always used in the corresponding biometric
framework. High level interface only contains the functionality, which
is needed to run biometric experiments. This, must have functionality,
is defined in the corresponding base classes and is discussed next.

First thing you need to do is to create a ``*.py`` file containing
your high level implementation, for example:
``bob/pad/face/database/replay.py`` for the Replay database. This file
must be placed into corresponding biometric framework, which in this
case is ``bob.pad.face`` package. The file **must** contain the
implementation of two classes:

-  ``<YourDatabaseName><Bio/Pad/Other>File``
-  ``<YourDatabaseName><Bio/Pad/Other>Database``

For example, the names of the above classes for the *Replay* database used in
the ``bob.pad.face`` framework are: ``ReplayPadFile`` and
``ReplayPadDatabase``.

Implementation of the ``*File`` class
---------------------------------------------------

First of all, the ``*File`` class must inherit from the **base file
class** of the corresponding biometric framework. An example:

-  ``*File`` class for the Replay database used in PAD (Presentation
   Attack Detection) experiments: ``class ReplayPadFile(PadFile):``
-  ``*File`` class for the Biowave V1 database used in verification
   experiments: ``class BiowaveV1BioFile(BioFile):``

Base class defines the elements, which must be implemented in the derived
class. For example, the implementation of ``ReplayPadFile`` class must
set the following elements of the base class: ``client_id``, ``path``,
``attack_type`` and ``file_id``. The corresponding high level
implementation of the ``ReplayPadFile`` class might look as follows:

.. code:: python

    import bob.bio.video

    from bob.pad.base.database import PadFile

    class ReplayPadFile(PadFile):
        def __init__(self, f):
            self.__f = f # here ``f`` is an instance of the File class defined in the low level database interface
            if f.is_real():
                attack_type = None
            else:
                attack_type = 'attack'
            super(ReplayPadFile, self).__init__(client_id=f.client, path=f.path,
                                                attack_type=attack_type, file_id=f.id)
        def load(self, directory=None, extension='.mov'):
            path = self.f.make_path(directory=directory, extension=extension)
            frame_selector = bob.bio.video.FrameSelector(selection_style = 'all')
            video_data = frame_selector(path)
            bbx_data = one_file.bbx(directory=directory)
            return_dictionary = {}
            return_dictionary["data"] = video_data
            return_dictionary["annotations"] = bbx_data
            return return_dictionary

Please, note, that in our case the ``ReplayPadFile`` also has a
``load()`` method. *Note: the load() method of the high level
``*File`` class is used by the preprocessor (a very first block in every
biometric pipeline) to read the data from the database.* Not all high
level database interfaces require this method, but let's try to
understand why ``ReplayPadFile`` class has it. The necessity to have
this method comes from the fact, that Replay database contains **video**
files, not images. To understand why ``load()`` method is needed in the
case of video-based database we need to take a look at the inheritance
structure of the class. For the ``ReplayPadFile`` class it looks as
follows:

-  ``ReplayPadFile`` -> ``bob.pad.base.database.PadFile`` ->
   ``bob.bio.base.database.BioFile`` -> ``bob.db.base.File``

Here the notation ``A`` -> ``B`` means ``A`` inherits from ``B``. Well,
the inheritance is pretty deep, but no need to worry about this. The
class of interest for us is ``bob.db.base.File`` containing the default
file managing methods, which might be overridden if necessary. One of
methods is ``load()`` **not** supporting video files by default. Since a
different behavior is desired, we need to override it in the high level
implementation of the ``*File`` class, ``ReplayPadFile`` in this case.
In this example the ``load()`` method returns the dictionary, which
contains the video frames, and annotations defining the face bounding
box in each frame. The preprocessor has to be "ready to deal" with that
type of input. With this, we are done configuring the high level
implementation of the ``*File`` class.

Implementation of the ``*Database`` class
---------------------------------------------------

The second unit to be implemented in HLDI is the ``*Database`` class.
First of all the ``*Database`` class must inherit from the **base
database class** of the corresponding biometric framework. An example:

-  ``*Database`` class for the Replay database used in PAD (Presentation
   Attack Detection) experiments:
   ``class ReplayPadDatabase(PadDatabase):``
-  ``*Database`` class for the Biowave V1 database used in verification
   experiments: ``class BiowaveV1BioDatabase(BioDatabase):``


Let's consider an example of the ``ReplayPadDatabase`` class. The implementation might look as follows, but don't dive into the code yet:

.. code:: python

    from bob.pad.base.database import PadDatabase

    class ReplayPadDatabase(PadDatabase):

        def __init__(
            self,
            all_files_options={},
            check_original_files_for_existence=False,
            original_directory=None,
            original_extension=None,
            # here I have said grandtest because this is the name of the default
            # protocol for this database
            protocol='grandtest',
            **kwargs):

            self.db = LowLevelDatabase()

            # Since the high level API expects different group names than what the low
            # level API offers, you need to convert them when necessary
            self.low_level_group_names = ('train', 'devel', 'test') # group names in the low-level database interface
            self.high_level_group_names = ('train', 'dev', 'eval') # names are expected to be like that in objects() function

            super(ReplayPadDatabase, self).__init__(
                'replay',
                all_files_options,
                check_original_files_for_existence,
                original_directory,
                original_extension,
                protocol,
                **kwargs)

        def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
            # Convert group names to low-level group names here.
            groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)
            files = self.db.objects(protocol=protocol, groups=groups, cls=purposes, **kwargs)
            files = [ReplayPadFile(f) for f in files]
            return files

        def annotations(self, file):
            """
            Do nothing. In this particular implementation the annotations are returned in the *File class above.
            """
            return None


Instead, let's try to understand why the implementation looks like this. Again, the methods to be implemented are defined by the corresponding base class of our ``*Database`` class.
In the case of PAD ``*Database`` the inheritance structure is as follows:

- ``ReplayPadDatabase`` -> ``bob.pad.base.database.PadDatabase`` -> ``bob.bio.base.database.BioDatabase`` -> ``bob.db.base.Database``

For the verification database the inheritance would be:

- ``bob.pad.base.database.PadDatabase`` -> ``bob.bio.base.database.BioDatabase`` -> ``bob.db.base.Database``

For other biometric experiments it might look differently.
In the given example the behavior of the ``ReplayPadDatabase`` class is defined by the ``bob.pad.base.database.PadDatabase`` base class, which sates that two methods must be implemented in the high level database implementation: ``objects()`` and ``annotations()``. The ``objects()`` method returns a list of instances of ``ReplayPadFile`` class. The ``annotations()`` method is empty, since the developer of the code decided to return the annotations in the ``*File`` class. Note: you are not obliged to do it that way, it's just a matter of taste.

At this point, having all necessary classes in place, we are done with implementation of the high level database interface!

Just a few small things have to be done to register our high level interface in the corresponding biometric framework.

- First, import your package in the ``__init__.py`` file located in the folder containing the implementation of HLDI: ``from .replay import ReplayPadDatabase``

- Next, create an instance of the ``*Database`` class with default configuration. For example, for the ``ReplayPadDatabase`` class used in ``bob.pad.face`` framework, the default configuration file ``/bob/pad/face/config/database/replay.py`` is as follows:

.. code:: python

    # The original_directory is taken from the .bob_bio_databases.txt file located in your home directory
    original_directory = "[YOUR_REPLAY_ATTACK_DIRECTORY]"
    original_extension = ".mov" # extension of the data files

    database = ReplayPadDatabase(
        protocol='grandtest',
        original_directory=original_directory,
        original_extension=original_extension,
        training_depends_on_protocol=True,
    )

- Finally, in the ``setup.py`` file of the corresponding biometric framework, add the entry pointing to your default configuration. In the case of observed PAD example the code is:

.. code:: python

    entry_points = {

        'bob.pad.database': [
            'replay = bob.pad.face.config.database.replay:database',
            ],

    },

That's it! Now we are ready to use our database in the corresponding biometric framework.