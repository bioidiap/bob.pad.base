#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 19 Aug 13:43:21 2015
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from bob.bio.base.database.Database import Database
import os

import antispoofing.utils.db


class DatabaseBobSpoof(Database):
    """This class can be used whenever you have a database that follows the Bob
    antispoofing database interface, which is defined in :py:class:`antispoofing.utils.db.Database`

    **Parameters:**

    database : derivative of :py:class:`antispoofing.utils.db.Database`
      The database instance that provides the actual interface, see :ref:`antispoofing_databases` for a list.

    all_files_options : dict
      Dictionary of options passed to the :py:meth:`antispoofing.utils.db.Database.objects` database query when retrieving all data.

    check_original_files_for_existence : bool
      Enables to test for the original data files when querying the database.

    kwargs : ``key=value`` pairs
      The arguments of the :py:class:`Database` base class constructor.

      .. note:: Usually, the ``name``, ``protocol`` keyword parameters of the base class constructor need to be specified.
    """

    def __init__(
            self,
            database,  # The bob database that is used
            all_files_options={},  # additional options for the database query that can be used to extract all files
            original_directory=None,  # the directory where the data files are located
            check_original_files_for_existence=False,
            **kwargs  # The default parameters of the base class
    ):

        Database.__init__(
            self,
            **kwargs
        )

        assert isinstance(database, antispoofing.utils.db.Database), \
            "Only databases derived from antispoofing.utils.db.Database are supported by this interface. " \
            "Please implement your own bob.bio.base.database.Database interface for anti-spoofing experiments."

        self.database = database
        if original_directory is None:
            self.original_directory = database.original_directory
        else:
            self.original_directory = original_directory

        self.all_files_options = all_files_options
        self.check_existence = check_original_files_for_existence

        self._kwargs = kwargs

    def set_protocol(self, protocol):
        """
        Sets the protocol for the database. The protocol can be specified via command line to spoof.py
        script with option -P
        :param protocol: name of the protocol
        :return: None
        """
        self.protocol = protocol
        self.database.set_kwargs({'protocol': protocol})

    def __str__(self):
        """__str__() -> info

        This function returns all parameters of this class (and its derived class).

        **Returns:**

        info : str
          A string containing the full information of all parameters of this (and the derived) class.
        """
        params = ", ".join(["%s=%s" % (key, value) for key, value in self._kwargs.items()])
        params += ", original_directory=%s" % (self.original_directory)
        if self.all_files_options: params += ", all_files_options=%s" % self.all_files_options

        return "%s(%s)" % (str(self.__class__), params)


    def replace_directories(self, replacements=None):
        """This helper function replaces the ``original_directory`` of the database with
        the directory read from the given replacement file.

        This function is provided for convenience, so that the database
        configuration files do not need to be modified.
        Instead, this function uses the given dictionary of replacements to change the original directory.

        The given ``replacements`` can be of type ``dict``, including all replacements,
        or a file name (as a ``str``), in which case the file is read.
        The structure of the file should be:

        .. code-block:: text

           # Comments starting with # and empty lines are ignored

           original/path/to/data = /path/to/your/data

        **Parameters:**

        replacements : dict or str
          A dictionary with replacements, or a name of a file to read the dictionary from.
          If the file name does not exist, no directories are replaced.
        """
        if replacements is None:
            return
        if isinstance(replacements, str):
            if not os.path.exists(replacements):
                return
            # Open the database replacement file and reads its content
            with open(replacements) as f:
                replacements = {}
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        splits = line.split("=")
                        assert len(splits) == 2
                        replacements[splits[0].strip()] = splits[1].strip()

        assert isinstance(replacements, dict)

        if self.original_directory in replacements:
            self.original_directory = replacements[self.original_directory]
            self.database.original_directory = self.original_directory


    def all_files(self, groups=('train', 'dev', 'eval')):
        """all_files(groups=('train', 'dev', 'eval')) -> files

        Returns all files of the database, respecting the current protocol.

        **Parameters:**

        groups : some of ``('train', 'dev', 'eval')`` or ``None``
          The groups to get the data for.
          If ``None``, data for all groups is returned.

        **Returns:**

        files : [:py:class:`antispoofing.utils.db.File`]
          The sorted and unique list of all files of the database.
        """
        realset = []
        attackset = []
        if 'train' in groups:
            real, attack = self.database.get_train_data()
            realset += real
            attackset += attack
        if 'dev' in groups:
            real, attack = self.database.get_devel_data()
            realset += real
            attackset += attack
        if 'eval' in groups:
            real, attack = self.database.get_test_data()
            realset += real
            attackset += attack
        return [realset, attackset]

    def training_files(self, step=None, arrange_by_client=False):
        """training_files(step = None, arrange_by_client = False) -> files

        Returns all training File objects
        This function needs to be implemented in derived class implementations.

        **Parameters:**
            The parameters are not applicable in this version of anti-spoofing experiments

        **Returns:**

        files : [:py:class:`File`] or [[:py:class:`File`]]
          The (arranged) list of files used for the training.
        """
        return self.database.get_train_data()

    def original_file_names(self, files):
        """original_file_names(files) -> paths

        Returns the full paths of the real and attack data of the given File objects.

        **Parameters:**

        files : [[:py:class:`antispoofing.utils.db.File`], [:py:class:`antispoofing.utils.db.File`]]
          The list of lists ([real, attack]]) of file object to retrieve the original data file names for.

        **Returns:**

        paths : [str]
          The paths extracted for the concatenated real+attack files, in the preserved order.
        """
        realfiles = files[0]
        attackfiles = files[1]
        realpaths = [file.make_path(directory=self.original_directory, extension=self.original_extension) for file in
                     realfiles]
        attackpaths = [file.make_path(directory=self.original_directory, extension=self.original_extension) for file in
                       attackfiles]
        return realpaths + attackpaths
