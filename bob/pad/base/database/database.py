#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date:   Tue May 17 12:09:22 CET 2016
#

import abc
import bob.bio.base
import bob.pad.base
from bob.bio.base.database import BioDatabase


class PadDatabase(BioDatabase):
    """This class represents the basic API for database access.
    Please use this class as a base class for your database access classes.
    Do not forget to call the constructor of this base class in your derived class.

    **Parameters:**

    name : str
    A unique name for the database.

    all_files_options : dict
    Dictionary of options passed to the second-level database query when retrieving all data.

    check_original_files_for_existence : bool
    Enables to test for the original data files when querying the database.

    original_directory : str
    The directory where the original data of the database are stored.

    original_extension : str
    The file name extension of the original data.

    protocol : str or ``None``
    The name of the protocol that defines the default experimental setup for this database.

    kwargs : ``key=value`` pairs
    The arguments of the :py:class:`bob.bio.base.database.BioDatabase` base class constructor.

    """

    def __init__(
            self,
            name,
            all_files_options={},  # additional options for the database query that can be used to extract all files
            check_original_files_for_existence=False,
            original_directory=None,
            original_extension=None,
            protocol='Default',
            **kwargs  # The rest of the default parameters of the base class
    ):
        super(PadDatabase, self).__init__(name=name, all_files_options=all_files_options, check_original_files_for_existence=check_original_files_for_existence, original_directory=original_directory, original_extension=original_extension, protocol=protocol, **kwargs)

    def original_file_names(self, files):
        """original_file_names(files) -> paths

        Returns the full paths of the real and attack data of the given PadFile objects.

        **Parameters:**

        files : [[:py:class:`bob.pad.base.database.PadFile`], [:py:class:`bob.pad.base.database.PadFile`]
          The list of lists ([real, attack]) of file object to retrieve the original data file names for.

        **Returns:**

        paths : [str] or [[str]]
          The paths extracted for the concatenated real+attack files, in the preserved order.
        """
        assert self.original_directory is not None
        assert self.original_extension is not None
        realfiles = files[0]
        attackfiles = files[1]
        realpaths = [file.make_path(directory=self.original_directory, extension=self.original_extension) for file in
                     realfiles]
        attackpaths = [file.make_path(directory=self.original_directory, extension=self.original_extension) for file in
                       attackfiles]
        return realpaths + attackpaths

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        """model_ids_with_protocol(groups = None, protocol = None, **kwargs) -> ids

            Client-based PAD is not implemented.
        """
        return []

    def annotations(self, file):
        """
        Annotations are not supported by PAD interface

        """
        return None

    @abc.abstractmethod
    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        """This function returns lists of File objects, which fulfill the given restrictions.

        Keyword parameters:

        groups : str or [str]
          The groups of which the clients should be returned.
          Usually, groups are one or more elements of ('train', 'dev', 'eval')

        protocol
          The protocol for which the clients should be retrieved.
          The protocol is dependent on your database.
          If you do not have protocols defined, just ignore this field.

        purposes : str or [str]
          The purposes for which File objects should be retrieved.
          Usually it is either 'real' or 'attack'.

        model_ids : [various type]
          This parameter is not suported in PAD databases yet
        """
        raise NotImplementedError("This function must be implemented in your derived class.")

    #################################################################
    ######### Methods to provide common functionality ###############
    #################################################################

    def all_files(self, groups=('train', 'dev', 'eval')):
        """all_files(groups=('train', 'dev', 'eval')) -> files

        Returns all files of the database, respecting the current protocol.
        The files can be limited using the ``all_files_options`` in the constructor.

        **Parameters:**

        groups : some of ``('train', 'dev', 'eval')`` or ``None``
          The groups to get the data for.

        **Returns:**

        files : [:py:class:`bob.pad.base.database.PadFile`]
          The sorted and unique list of all files of the database.
        """
        realset = self.sort(self.objects(protocol=self.protocol, groups=groups, purposes='real', **self.all_files_options))
        attackset = self.sort(self.objects(protocol=self.protocol, groups=groups, purposes='attack', **self.all_files_options))
        return [realset, attackset]

    def training_files(self, step=None, arrange_by_client=False):
        """training_files(step = None, arrange_by_client = False) -> files

        Returns all training File objects
        This function needs to be implemented in derived class implementations.

        **Parameters:**
            The parameters are not applicable in this version of anti-spoofing experiments

        **Returns:**

        files : [:py:class:`bob.pad.base.database.PadFile`] or [[:py:class:`bob.pad.base.database.PadFile`]]
          The (arranged) list of files used for the training.
        """

        return self.all_files(groups=('train',))
