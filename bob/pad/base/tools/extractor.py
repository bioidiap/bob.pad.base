#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 19 Aug 13:43:21 2015
#


import bob.io.base
import os

import logging

logger = logging.getLogger("bob.pad.base")

from .FileSelector import FileSelector
from bob.bio.base import utils
from .preprocessor import read_preprocessed_data
from bob.bio.base.tools.extractor import read_features


def train_extractor(extractor, preprocessor, allow_missing_files=False, force=False):
    """Trains the feature extractor using preprocessed data of the ``'train'`` group,
    if the feature extractor requires training.

    This function should only be called, when the ``extractor`` actually requires training.
    The given ``extractor`` is trained using preprocessed data.
    It writes the extractor to the file specified by the :py:class:`bob.pad.base.tools.FileSelector`.
    By default, if the target file already exist, it is not re-created.

    **Parameters:**

    extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
      The extractor to be trained.

    preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
      The preprocessor, used for reading the preprocessed data.

    allow_missing_files : bool
      If set to ``True``, preprocessed data files that are not found are silently ignored during training.

    force : bool
      If given, the extractor file is regenerated, even if it already exists.
    """
    if not extractor.requires_training:
        logger.warn(
            "The train_extractor function should not have been called, since the extractor does not need training.")
        return

    # the file selector object
    fs = FileSelector.instance()
    # the file to write
    if utils.check_file(fs.extractor_file, force,
                        extractor.min_extractor_file_size):
        logger.info("- Extraction: extractor '%s' already exists.",
                    fs.extractor_file)
    else:
        bob.io.base.create_directories_safe(os.path.dirname(fs.extractor_file))
        # read training files
        train_files = fs.training_list(
            'preprocessed', 'train_extractor', combined=~extractor.split_training_data_by_client)
        train_data = read_preprocessed_data(
            train_files, preprocessor, extractor.split_training_data_by_client, allow_missing_files)
        if extractor.split_training_data_by_client:
            logger.info("- Extraction: training extractor '%s' using %d classes:",
                        fs.extractor_file, len(train_files))
        else:
            logger.info("- Extraction: training extractor '%s' using %d training files:",
                        fs.extractor_file, len(train_files))
        # train model
        extractor.train(train_data, fs.extractor_file)


def extract(extractor, preprocessor, groups=None, indices=None, allow_missing_files=False, force=False):
    """Extracts features from the preprocessed data using the given extractor.

    The given ``extractor`` is used to extract all features required for the current experiment.
    It writes the extracted data into the directory specified by the :py:class:`bob.pad.base.tools.FileSelector`.
    By default, if target files already exist, they are not re-created.

    The preprocessor is only used to load the data in a coherent way.

    **Parameters:**

    extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
      The extractor, used for extracting and writing the features.

    preprocessor : py:class:`bob.bio.base.preprocessor.Preprocessor` or derived
      The preprocessor, used for reading the preprocessed data.

    groups : some of ``('train', 'dev', 'eval')`` or ``None``
      The list of groups, for which the data should be extracted.

    indices : (int, int) or None
      If specified, only the features for the given index range ``range(begin, end)`` should be extracted.
      This is usually given, when parallel threads are executed.

    allow_missing_files : bool
      If set to ``True``, preprocessed data files that are not found are silently ignored.

    force : bool
      If given, files are regenerated, even if they already exist.
    """
    # the file selector object
    fs = FileSelector.instance()
    extractor.load(fs.extractor_file)
    data_files = fs.preprocessed_data_list(groups=groups)
    feature_files = fs.feature_list(groups=groups)

    # select a subset of indices to iterate
    if indices is not None:
        index_range = range(indices[0], indices[1])
        logger.info("- Extraction: splitting of index range %s" % str(indices))
    else:
        index_range = range(len(data_files))

    logger.info("- Extraction: extracting %d features from directory '%s' to directory '%s'", len(index_range),
                fs.directories['preprocessed'], fs.directories['extracted'])
    for i in index_range:
        data_file = data_files[i]
        feature_file = feature_files[i]

        if not os.path.exists(data_file) and preprocessor.writes_data:
            if allow_missing_files:
                logger.debug(
                    "... Cannot find preprocessed data file %s; skipping", data_file)
                continue
            else:
                logger.error(
                    "Cannot find preprocessed data file %s", data_file)

        if not utils.check_file(feature_file, force,
                                extractor.min_feature_file_size):
            logger.debug(
                "... Extracting features for data file '%s'", data_file)
            # create output directory before reading the data file (is
            # sometimes required, when relative directories are specified,
            # especially, including a .. somewhere)
            bob.io.base.create_directories_safe(os.path.dirname(feature_file))
            # load data
            data = preprocessor.read_data(data_file)
            # extract feature
            feature = extractor(data)

            if feature is None:
                if allow_missing_files:
                    logger.debug(
                        "... Feature extraction for data file %s failed; skipping", data_file)
                    continue
                else:
                    raise RuntimeError(
                        "Feature extraction  of file '%s' was not successful", data_file)

            # write feature
            extractor.write_feature(feature, feature_file)
        else:
            logger.debug(
                "... Skipping preprocessed data '%s' since feature file '%s' exists", data_file, feature_file)
