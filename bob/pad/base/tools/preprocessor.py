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

import bob.io.base
import os

import logging

logger = logging.getLogger("bob.pad.base")

from .FileSelector import FileSelector
from bob.bio.base import utils


def preprocess(preprocessor, groups=None, indices=None, force=False):
    """Preprocesses the original data of the database with the given preprocessor.

    The given ``preprocessor`` is used to preprocess all data required for the current experiment.
    It writes the preprocessed data into the directory specified by the :py:class:`bob.pad.base.tools.FileSelector`.
    By default, if target files already exist, they are not re-created.

    **Parameters:**

    preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived.
      The preprocessor, which should be applied to all data.

    groups : some of ``('train', 'dev', 'eval')`` or ``None``
      The list of groups, for which the data should be preprocessed.

    indices : (int, int) or None
      If specified, only the data for the given index range ``range(begin, end)`` should be preprocessed.
      This is usually given, when parallel threads are executed.

    force : bool
      If given, files are regenerated, even if they already exist.
    """
    # the file selector object
    fs = FileSelector.instance()

    # get the file lists
    data_files, original_directory, original_extension = fs.original_data_list_files(groups=groups)
    preprocessed_data_files = fs.preprocessed_data_list(groups=groups)
    print("len of data files: %s" %(str(len(data_files))))
    print("len of preprocessed data files (paths): %s" %(str(len(preprocessed_data_files))))

    # select a subset of keys to iterate
    if indices is not None:
        index_range = range(indices[0], indices[1])
        logger.info("- Preprocessing: splitting of index range %s", str(indices))
    else:
        index_range = range(len(data_files))

    logger.info("- Preprocessing: processing %d data files from directory '%s' to directory '%s'", len(index_range),
                fs.directories['original'], fs.directories['preprocessed'])

    # iterate over the selected files
    for i in index_range:
        preprocessed_data_file = str(preprocessed_data_files[i])
        file_object = data_files[i]
        file_name = file_object.make_path(original_directory, original_extension)

        # check for existence
        if not utils.check_file(preprocessed_data_file, force, 1000):
            logger.info("... Processing original data file '%s'", file_name)
            data = preprocessor.read_original_data(file_object, original_directory, original_extension)
            # create output directory before reading the data file (is sometimes required, when relative directories are specified, especially, including a .. somewhere)
            bob.io.base.create_directories_safe(os.path.dirname(preprocessed_data_file))

            # call the preprocessor
            preprocessed_data = preprocessor(data, None)
            if preprocessed_data is None:
                logger.error("Preprocessing of file '%s' was not successful", file_name)
                continue

            # write the data
            preprocessor.write_data(preprocessed_data, preprocessed_data_file)


def read_preprocessed_data(file_names, preprocessor):
    """read_preprocessed_data(file_names, preprocessor, split_by_client = False) -> preprocessed

    Reads the preprocessed data from ``file_names`` using the given preprocessor.
    If ``split_by_client`` is set to ``True``, it is assumed that the ``file_names`` are already sorted by client.

    **Parameters:**

    file_names : [str] or [[str]]
      A list of names of files to be read.
      If ``split_by_client = True``, file names are supposed to be split into groups.

    preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
      The preprocessor, which can read the preprocessed data.

    **Returns:**

    preprocessed : [object] or [[object]]
      The list of preprocessed data, in the same order as in the ``file_names``.
    """
    return [preprocessor.read_data(str(f)) for f in file_names]
