#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#

from __future__ import print_function

import numpy as np

from bob.io.base.test_utils import datafile
from bob.io.base import load

import bob.io.image  # for image loading functionality
import bob.bio.video
import bob.pad.base

from bob.pad.base.algorithm import SVM

import random

from bob.pad.base.utils import convert_array_to_list_of_frame_cont, convert_frame_cont_to_array

def test_video_svm_pad_algorithm():
    """
    Test the VideoSvmPadAlgorithm algorithm.
    """

    random.seed(7)

    N = 20000
    mu = 1
    sigma = 1
    real_array = np.transpose(
        np.vstack([[random.gauss(mu, sigma) for _ in range(N)],
                   [random.gauss(mu, sigma) for _ in range(N)]]))

    mu = 5
    sigma = 1
    attack_array = np.transpose(
        np.vstack([[random.gauss(mu, sigma) for _ in range(N)],
                   [random.gauss(mu, sigma) for _ in range(N)]]))

    real = convert_array_to_list_of_frame_cont(real_array)
    attack = convert_array_to_list_of_frame_cont(attack_array)

    training_features = [real, attack]

    MACHINE_TYPE = 'C_SVC'
    KERNEL_TYPE = 'RBF'
    N_SAMPLES = 1000
    TRAINER_GRID_SEARCH_PARAMS = {'cost': [1], 'gamma': [0.5, 1]}
    MEAN_STD_NORM_FLAG = True  # enable mean-std normalization
    FRAME_LEVEL_SCORES_FLAG = True  # one score per frame(!) in this case

    algorithm = SVM(
        machine_type=MACHINE_TYPE,
        kernel_type=KERNEL_TYPE,
        n_samples=N_SAMPLES,
        trainer_grid_search_params=TRAINER_GRID_SEARCH_PARAMS,
        mean_std_norm_flag=MEAN_STD_NORM_FLAG,
        frame_level_scores_flag=FRAME_LEVEL_SCORES_FLAG)

    machine = algorithm.train_svm(
        training_features=training_features,
        n_samples=algorithm.n_samples,
        machine_type=algorithm.machine_type,
        kernel_type=algorithm.kernel_type,
        trainer_grid_search_params=algorithm.trainer_grid_search_params,
        mean_std_norm_flag=algorithm.mean_std_norm_flag,
        projector_file="",
        save_debug_data_flag=False)

    assert machine.n_support_vectors == [148, 150]
    assert machine.gamma == 0.5

    real_sample = convert_frame_cont_to_array(real[0])

    prob = machine.predict_class_and_probabilities(real_sample)[1]

    assert prob[0, 0] > prob[0, 1]

    precision = algorithm.comp_prediction_precision(machine, real_array,
                                                    attack_array)

    assert precision > 0.99