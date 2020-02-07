#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Anjith George
"""

# ==============================================================================

from .ScikitClassifier import ScikitClassifier

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler


class OneClassGMM(ScikitClassifier):
    """
    This class is designed to train a OneClassGMM based PAD system. The OneClassGMM is trained
    using data of one class (real class) only. The procedure is the following:

    1. First, the training data is mean-std normalized using mean and std of the
       real class only.

    2. Second, the OneClassGMM with ``n_components`` Gaussians is trained using samples
       of the real class.

    3. The input features are next classified using pre-trained OneClassGMM machine.

    **Parameters:**

    ``n_components`` : :py:class:`int`
        Number of Gaussians in the OneClassGMM. Default: 1 .

    ``random_state`` : :py:class:`int`
        A seed for the random number generator used in the initialization of
        the OneClassGMM. Default: 3 .

    ``frame_level_scores_flag`` : :py:class:`bool`
        Return scores for each frame individually if True. Otherwise, return a
        single score per video. Default: False.
    """

    def __init__(self,
                 n_components=1,
                 random_state=3,
                 frame_level_scores_flag=False,
                 covariance_type='full',
                 reg_covar=1e-06,
                 ):

        ScikitClassifier.__init__(self,
                                  clf=GaussianMixture(n_components=n_components,
                                                      random_state=random_state,
                                                      covariance_type=covariance_type,
                                                      reg_covar=reg_covar),
                                  scaler=StandardScaler(),
                                  frame_level_scores_flag=frame_level_scores_flag,
                                  norm_on_bonafide=True,
                                  one_class=True)
