#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np
import bob.bio.video

import itertools


def convert_frame_cont_to_array(frame_container):
    """
    This function converts a single Frame Container into an array of features.
    The rows are samples, the columns are features.

    **Parameters:**

    ``frame_container`` : object
        A Frame Container conteining the features of an individual,
        see ``bob.bio.video.utils.FrameContainer``.

    **Returns:**

    ``features_array`` : 2D :py:class:`numpy.ndarray`
        An array containing features for all frames.
        The rows are samples, the columns are features.
    """

    feature_vectors = []

    frame_dictionary = {}

    for frame in frame_container:
        frame_dictionary[frame[0]] = frame[1]

    for idx, _ in enumerate(frame_container):
        # Frames are stored in a mixed order, therefore we get them using incrementing frame index:
        feature_vectors.append(frame_dictionary[str(idx)])

    features_array = np.vstack(feature_vectors)

    return features_array


def convert_and_prepare_features(features):
    """
    This function converts a list or a frame container of features into a 2D array of features.
    If the input is a list of frame containers, features from different frame containers (individuals)
    are concatenated into the same list. This list is then converted to an array. The rows are samples,
    the columns are features.

    **Parameters:**

    ``features`` : [2D :py:class:`numpy.ndarray`] or [FrameContainer]
        A list or 2D feature arrays or a list of Frame Containers, see ``bob.bio.video.utils.FrameContainer``.
        Each frame Container contains feature vectors for the particular individual/person.

    **Returns:**

    ``features_array`` : 2D :py:class:`numpy.ndarray`
        An array containing features for all samples and frames.
    """

    if isinstance(
            features[0],
            bob.bio.video.FrameContainer):  # if FrameContainer convert to 2D numpy array
        return convert_list_of_frame_cont_to_array(features)
    else:
        return np.vstack(features)


def convert_list_of_frame_cont_to_array(frame_containers):
    """
    This function converts a list of Frame containers into an array of features.
    Features from different frame containers (individuals) are concatenated into the
    same list. This list is then converted to an array. The rows are samples,
    the columns are features.

    **Parameters:**

    ``frame_containers`` : [FrameContainer]
        A list of Frame Containers, , see ``bob.bio.video.utils.FrameContainer``.
        Each frame Container contains feature vectors for the particular individual/person.

    **Returns:**

    ``features_array`` : 2D :py:class:`numpy.ndarray`
        An array containing features for all frames of all individuals.
    """

    feature_vectors = []

    for frame_container in frame_containers:
        video_features_array = convert_frame_cont_to_array(
            frame_container)

        feature_vectors.append(video_features_array)

    features_array = np.vstack(feature_vectors)

    return features_array


def combinations(input_dict):
    """
    Obtain all possible key-value combinations in the input dictionary
    containing list values.

    **Parameters:**

    ``input_dict`` : :py:class:`dict`
        Input dictionary with list values.

    **Returns:**

    ``combinations`` : [:py:class:`dict`]
        A list of dictionaries containing the combinations.
    """

    varNames = sorted(input_dict)

    combinations = [
        dict(zip(varNames, prod))
        for prod in itertools.product(*(input_dict[varName]
                                        for varName in varNames))
        ]

    return combinations


def select_uniform_data_subset(features, n_samples):
    """
    Uniformly select N samples/feature vectors from the input array of samples.
    The rows in the input array are samples. The columns are features.

    **Parameters:**

    ``features`` : 2D :py:class:`numpy.ndarray`
        Input array with feature vectors. The rows are samples, columns are features.

    ``n_samples`` : :py:class:`int`
        The number of samples to be selected uniformly from the input array of features.

    **Returns:**

    ``features_subset`` : 2D :py:class:`numpy.ndarray`
        Selected subset of features.
    """

    if features.shape[0] <= n_samples:

        features_subset = features

    else:

        uniform_step = np.int(features.shape[0] / n_samples)

        features_subset = features[0:np.int(uniform_step * n_samples):
        uniform_step, :]

    return features_subset


def select_quasi_uniform_data_subset(features, n_samples):
    """
    Select quasi uniformly N samples/feature vectors from the input array of samples.
    The rows in the input array are samples. The columns are features.
    Use this function if n_samples is close to the number of samples.

    **Parameters:**

    ``features`` : 2D :py:class:`numpy.ndarray`
        Input array with feature vectors. The rows are samples, columns are features.

    ``n_samples`` : :py:class:`int`
        The number of samples to be selected uniformly from the input array of features.

    **Returns:**

    ``features_subset`` : 2D :py:class:`numpy.ndarray`
        Selected subset of features.
    """

    if features.shape[0] <= n_samples:

        features_subset = features

    else:

        uniform_step = (1.0 * features.shape[0]) / n_samples

        element_num_list = range(0, n_samples)

        idx = [np.int(uniform_step * item) for item in element_num_list]

        features_subset = features[idx, :]

    return features_subset


def convert_array_to_list_of_frame_cont(data):
    """
    Convert an input 2D array to a list of FrameContainers.

    **Parameters:**

    ``data`` : 2D :py:class:`numpy.ndarray`
        Input data array of the dimensionality (N_samples X N_features ).

        **Returns:**

    ``frame_container_list`` : [FrameContainer]
        A list of FrameContainers, see ``bob.bio.video.utils.FrameContainer``
        for further details. Each frame container contains one feature vector.
    """

    frame_container_list = []

    for idx, vec in enumerate(data):
        frame_container = bob.bio.video.FrameContainer(
        )  # initialize the FrameContainer

        frame_container.add(0, vec)

        frame_container_list.append(
            frame_container)  # add current frame to FrameContainer

    return frame_container_list


def mean_std_normalize(features,
                       features_mean=None,
                       features_std=None):
    """
    The features in the input 2D array are mean-std normalized.
    The rows are samples, the columns are features. If ``features_mean``
    and ``features_std`` are provided, then these vectors will be used for
    normalization. Otherwise, the mean and std of the features is
    computed on the fly.

    **Parameters:**

    ``features`` : 2D :py:class:`numpy.ndarray`
        Array of features to be normalized.

    ``features_mean`` : 1D :py:class:`numpy.ndarray`
        Mean of the features. Default: None.

    ``features_std`` : 2D :py:class:`numpy.ndarray`
        Standart deviation of the features. Default: None.

    **Returns:**

    ``features_norm`` : 2D :py:class:`numpy.ndarray`
        Normalized array of features.

    ``features_mean`` : 1D :py:class:`numpy.ndarray`
        Mean of the features.

    ``features_std`` : 1D :py:class:`numpy.ndarray`
        Standart deviation of the features.
    """

    features = np.copy(features)

    # Compute mean and std if not given:
    if features_mean is None:
        features_mean = np.mean(features, axis=0)

        features_std = np.std(features, axis=0)

    row_norm_list = []

    for row in features:  # row is a sample

        row_norm = (row - features_mean) / features_std

        row_norm_list.append(row_norm)

    features_norm = np.vstack(row_norm_list)

    return features_norm, features_mean, features_std