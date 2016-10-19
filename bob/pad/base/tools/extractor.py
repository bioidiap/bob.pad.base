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


def extract(extractor, preprocessor, groups=None, indices=None, force=False):
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

    force : bool
      If given, files are regenerated, even if they already exist.
    """
    # the file selector object
    fs = FileSelector.instance()
    data_files = fs.preprocessed_data_list(groups=groups)
    feature_files = fs.feature_list(groups=groups)

    # select a subset of indices to iterate
    if indices != None:
        index_range = range(indices[0], indices[1])
        logger.info("- Extraction: splitting of index range %s" % str(indices))
    else:
        index_range = range(len(data_files))

    logger.info("- Extraction: extracting %d features from directory '%s' to directory '%s'", len(index_range),
                fs.directories['preprocessed'], fs.directories['extracted'])
    for i in index_range:
        data_file = str(data_files[i])
        feature_file = str(feature_files[i])

        if not utils.check_file(feature_file, force, 1000):
            # load data
            data = preprocessor.read_data(data_file)
            # extract feature
            try:
                logger.info("- Extraction: extracting from file: %s", data_file)
                feature = extractor(data)
            except ValueError:
                logger.warn("WARNING: empty data in file %s", data_file)
                feature = 0
            # write feature
            if feature is not None:
                bob.io.base.create_directories_safe(os.path.dirname(feature_file))
                extractor.write_feature(feature, feature_file)


def read_features(file_names, extractor):
    """read_features(file_names, extractor) -> extracted

    Reads the extracted features from ``file_names`` using the given ``extractor``.

    **Parameters:**

    file_names : [[str], [str]]
      A list of lists of file names (real, attack) to be read.

    extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
      The extractor, used for reading the extracted features.

    **Returns:**

    extracted : [object] or [[object]]
      The list of extracted features, in the same order as in the ``file_names``.
    """
    real_files = file_names[0]
    attack_files = file_names[1]
    return [[extractor.read_feature(str(f)) for f in real_files],
            [extractor.read_feature(str(f)) for f in attack_files]]
