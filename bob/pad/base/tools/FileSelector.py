#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
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


import os
from bob.bio.base import utils


@utils.Singleton
class FileSelector:
    """This class provides shortcuts for selecting different files for different stages of the snti-spoofing process.

    It communicates with the database and provides lists of file names for all steps of the tool chain.

    **Parameters:**

    database : :py:class:`antispoofing.utils.db` or derived.
      The database object that provides the list of files.

    preprocessed_directory : str
      The directory, where preprocessed data should be written to.

    extracted_directory : str
      The directory, where extracted features should be written to.

    projector_file : str
      The filename, where the projector should be written to (if any).

    projected_directory : str
      The directory, where projected features should be written to (if required).

    score_directories : (str, str)
      The directories, where score files for no-norm should be written to.


    default_extension : str
      The default extension of all intermediate files.

    compressed_extension : str
      The extension for writing compressed score files.
      By default, no compression is performed.

    """

    def __init__(
            self,
            database,
            preprocessed_directory,
            extracted_directory,
            projector_file,
            projected_directory,
            score_directories,
            default_extension='.hdf5',
            compressed_extension=''
    ):

        """Initialize the file selector object with the current configuration."""
        self.database = database
        self.projector_file = projector_file

        self.score_directories = score_directories
        self.default_extension = default_extension
        self.compressed_extension = compressed_extension

        self.directories = {
            'original': database.original_directory,
            'preprocessed': preprocessed_directory,
            'extracted': extracted_directory,
            'projected': projected_directory
        }

    def get_paths(self, files, directory_type=None, combined=True):
        """Returns the lists of file names [real, attacks] for the given File objects."""
        try:
            directory = self.directories[directory_type]
        except KeyError:
            raise ValueError("The given directory type '%s' is not supported." % directory_type)

        # only one set of files
        if len(files) != 2:
            return self.database.file_names(files, directory, self.default_extension)
        realfiles = files[0]
        attackfiles = files[1]
        realpaths = self.database.file_names(realfiles, directory, self.default_extension)
        attackpaths = self.database.file_names(attackfiles, directory, self.default_extension)
        if combined:
            return realpaths + attackpaths
        else:
            return [realpaths, attackpaths]

    # List of files that will be used for all files
    def original_data_list(self, groups=None):
        """Returns the tuple of lists of original (real, attack) data that can be used for preprocessing."""
        return self.database.original_file_names(self.database.all_files(groups=groups))

    def preprocessed_data_list(self, groups=None):
        """Returns the tuple of lists (real, attacks) of preprocessed data files."""
        return self.get_paths(self.database.all_files(groups=groups), "preprocessed")

    def feature_list(self, groups=None):
        """Returns the tuple of lists (real, attacks) of extracted feature files."""
        return self.get_paths(self.database.all_files(groups=groups), "extracted")

    def projected_list(self, groups=None):
        """Returns the tuple of lists (real, attacks) of projected feature files."""
        return self.get_paths(self.database.all_files(groups=groups), "projected")

    # Training lists
    def training_list(self, directory_type, step):
        """Returns the tuple of lists (real, attacks) of features that should be used for projector training.
        The directory_type might be any of 'preprocessed', 'extracted', or 'projected'.
        The step might by any of 'train_extractor', 'train_projector', or 'train_enroller'.
        """
        return self.get_paths(self.database.training_files(step), directory_type, False)

    def toscore_objects(self, group):
        """Returns the File objects used to compute the raw scores."""
        # get the test files for all models
        return self.database.all_files(groups=(group,))

    def score_file_combined(self, group):
        """Returns the resulting score text file for the given group."""
        no_norm_dir = self.score_directories[0]
        return os.path.join(no_norm_dir, "scores-" + group) + self.compressed_extension

    def score_file_for_type(self, group, obj_type):
        """Returns the resulting score text file for the given group."""
        no_norm_dir = self.score_directories[0]
        return os.path.join(no_norm_dir, "scores-" + group + "-" + obj_type) + self.compressed_extension

