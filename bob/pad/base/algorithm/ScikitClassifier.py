#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Anjith George
"""

#==============================================================================

from bob.pad.base.algorithm import Algorithm

from bob.bio.video.utils import FrameContainer

import numpy as np

import bob.io.base

import pickle

from bob.pad.base.utils import convert_frame_cont_to_array, convert_list_of_frame_cont_to_array

#==============================================================================


class ScikitClassifier(Algorithm):
    """
    This class is designed to train any generic scikit-learn binary classifier given Frame Containers
    with features of real and attack classes. The procedure is the following:

    1. First, the input data is normalized using the scaler class.

    2. Second, the Scikit Algorithm is trained on normalized
       input features.

    3. The input features are next classified using pre-trained Scikit model.

    **Parameters:**

    ``clf`` : :py:class:`object`
        An sklearn binary classifier class, which is initialized in the config file.

    ``scaler`` : :py:class:`object`
        An sklearn scaler class which is initialized in the config file.

    ``frame_level_scores_flag`` : :py:class:`bool`
        Return scores for each frame individually if True. Otherwise, return a
        single score per video. Default: ``False``.

    ``subsample_train_data_flag`` : :py:class:`bool`
        Uniformly subsample the training data if ``True``. Default: ``False``.

    ``subsampling_step`` : :py:class:`int`
        Training data subsampling step, only valid is
        ``subsample_train_data_flag = True``. Default: 10 .

    ``subsample_videos_flag`` : :py:class:`bool`
        Uniformly subsample the training videos if ``True``. Default: ``False``.

    ``video_subsampling_step`` : :py:class:`int`
        Training videos subsampling step, only valid is
        ``subsample_videos_flag = True``. Default: 3 .
    ``norm_on_bonafide`` : :py:class:`bool`
        If set to `True` the normalizayion parameters are found from bonafide samples
        only. If set to `False`, both bonafide and attacks will be used to find normalization parameters.  
    """

    def __init__(self,
                 clf=None,
                 scaler=None,
                 frame_level_scores_flag=False,
                 subsample_train_data_flag=False,
                 subsampling_step=10,
                 subsample_videos_flag=False,
                 video_subsampling_step=3,
                 norm_on_bonafide=True, one_class=False):

        Algorithm.__init__(self,
                           clf=clf,
                           scaler=scaler,
                           frame_level_scores_flag=frame_level_scores_flag,
                           subsample_train_data_flag=subsample_train_data_flag,
                           subsampling_step=subsampling_step,
                           subsample_videos_flag=subsample_videos_flag,
                           video_subsampling_step=video_subsampling_step,
                           performs_projection=True,
                           requires_projector_training=True, 
                           norm_on_bonafide=norm_on_bonafide,
                           one_class=one_class)

        self.clf = clf

        self.scaler = scaler

        self.frame_level_scores_flag = frame_level_scores_flag

        self.subsample_train_data_flag = subsample_train_data_flag

        self.subsampling_step = subsampling_step

        self.subsample_videos_flag = subsample_videos_flag

        self.video_subsampling_step = video_subsampling_step

        self.norm_on_bonafide = norm_on_bonafide

        self.one_class = one_class


    #==========================================================================
    def _normalize(self, features, train=False):
        """
        The features in the input 2D array are normalized.
        The rows are samples, the columns are features. If train==True then 
        the scaler is trained, else the trained scaler is used for the normalization.

        **Parameters:**

        ``features`` : 2D :py:class:`numpy.ndarray`
            Array of features to be normalized.

        **Returns:**

        ``features_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized array of features.

        """

        if self.scaler is not None:
            if train:
                self.scaler.fit(features)

            features_norm = self.scaler.transform(features)
        else:
            features_norm=features.copy()

        return features_norm

    #==========================================================================
    def norm_train_data(self, real, attack):
        """
        Mean-std normalization of input data arrays. The mean and std normalizers
        are computed using real class only.

        **Parameters:**

        ``real`` : 2D :py:class:`numpy.ndarray`
            Training features for the real class.

        ``attack`` : 2D :py:class:`numpy.ndarray`
            Training features for the attack class.

        **Returns:**

        ``real_norm`` : 2D :py:class:`numpy.ndarray`
            Mean-std normalized training features for the real class.

        ``attack_norm`` : 2D :py:class:`numpy.ndarray`
            Mean-std normalized training features for the attack class.
        """

        if  self.norm_on_bonafide: # normalization parameters calculated from bonafide only

            real_norm = self._normalize(real, train=True)

            attack_norm = self._normalize(attack, train=False)

        else:

            all_data=np.vstack([real, attack])

            _ = self._normalize(all_data, train=True)

            real_norm = self._normalize(real, train=False)

            attack_norm = self._normalize(attack, train=False)

        return real_norm, attack_norm

    #==========================================================================
    def train_clf(self, real, attack):
        """
        Train GENERIC classifier given real and attack classes. Prior to training
        the data is mean-std normalized.

        **Parameters:**

        ``real`` : 2D :py:class:`numpy.ndarray`
            Training features for the real class.

        ``attack`` : 2D :py:class:`numpy.ndarray`
            Training features for the attack class.

        **Returns:**

        ``machine`` : object
            A trained GENERIC machine.
        """

        if self.one_class:
            assert(self.norm_on_bonafide==True)


        real, attack = self.norm_train_data(real, attack)
        # real and attack - are now mean-std normalized

        assert(self.clf is not None)

        if self.one_class:

            X=real.copy()

            Y=np.ones(len(real))

            self.clf.fit(X)

        else:
            X = np.vstack([real, attack])

            Y = np.hstack([np.ones(len(real)), np.zeros(len(attack))])

            self.clf.fit(X, Y)

        return True

    #==========================================================================
    def save_clf_and_mean_std(self, projector_file):
        """
        Saves the GENERIC machine, scaling parameters to a '.obj' file.
        The absolute name of the file is specified in ``projector_file`` string.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            Absolute name of the file to save the data to, as returned by
            ``bob.pad.base`` framework.

        ``machine`` : object
            The GENERIC machine to be saved. As returned by sklearn
            modules.


        """

        # Dumping Machine
        projector_file_n = projector_file[:-5]+'_skmodel.obj'
        with open(projector_file_n, 'wb') as fp:
            pickle.dump(self.clf, fp)

        # Dumping the scaler

        scaler_file_n = projector_file[:-5]+'_scaler.obj'
        with open(scaler_file_n, 'wb') as fp:
            pickle.dump(self.scaler, fp)

    #==========================================================================
    def subsample_train_videos(self, training_features, step):
        """
        Uniformly select subset of frmae containes from the input list

        **Parameters:**

        ``training_features`` : [FrameContainer]
            A list of FrameContainers

        ``step`` : :py:class:`int`
            Data selection step.

        **Returns:**

        ``training_features_subset`` : [FrameContainer]
            A list with selected FrameContainers
        """

        indexes = range(0, len(training_features), step)

        training_features_subset = [training_features[x] for x in indexes]

        return training_features_subset

    #==========================================================================
    def train_projector(self, training_features, projector_file):
        """
        Train GENERIC for feature projection and save them to files.
        The ``requires_projector_training = True`` flag must be set to True
        to enable this function.

        **Parameters:**

        ``training_features`` : [[FrameContainer], [FrameContainer]]
            A list containing two elements: [0] - a list of Frame Containers with
            feature vectors for the real class; [1] - a list of Frame Containers with
            feature vectors for the attack class.

        ``projector_file`` : :py:class:`str`
            The file to save the trained projector to, as returned by the
            ``bob.pad.base`` framework.
        """

        # training_features[0] - training features for the REAL class.
        # training_features[1] - training features for the ATTACK class.

        if self.subsample_videos_flag:  # subsample videos of the real class

            real = convert_list_of_frame_cont_to_array(self.subsample_train_videos(training_features[0], self.video_subsampling_step))  # output is array

        else:

            real = convert_list_of_frame_cont_to_array(training_features[0])  # output is array

        if self.subsample_train_data_flag:

            real = real[range(0, len(real), self.subsampling_step), :]

        if self.subsample_videos_flag:  # subsample videos of the real class

            attack = convert_list_of_frame_cont_to_array(self.subsample_train_videos(training_features[1], self.video_subsampling_step))  # output is array

        else:

            attack = convert_list_of_frame_cont_to_array(training_features[1])  # output is array

        if self.subsample_train_data_flag:

            attack = attack[range(0, len(attack), self.subsampling_step), :]

        # Train the GENERIC machine and get normalizers:
        self.train_clf(real=real, attack=attack)

        # Save the GENERIC machine and normalizers:
        self.save_clf_and_mean_std(projector_file)

    #==========================================================================
    def load_clf_and_mean_std(self, projector_file):
        """
        Loads the machine, features mean and std from the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            Absolute name of the file to load the trained projector from, as
            returned by ``bob.pad.base`` framework.

        **Returns:**

        ``machine`` : object
            The loaded GENERIC machine. As returned by sklearn.linear_model module.

        """

        projector_file_n = projector_file[:-5]+'_skmodel.obj'

        # Load the params of the machine:

        with open(projector_file_n, 'rb') as fp:
            self.clf = pickle.load(fp)

        scaler_file_n = projector_file[:-5]+'_scaler.obj'

        # Load parameters of the scaler:

        with open(scaler_file_n, 'rb') as fp:
            self.scaler = pickle.load(fp)


    #==========================================================================
    def load_projector(self, projector_file):
        """
        Loads the machine, features mean and std from the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        This function sets the arguments ``self.clf``, with loaded machines.


        Please register `performs_projection = True` in the constructor to
        enable this function.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            The file to read the projector from, as returned by the
            ``bob.pad.base`` framework. In this class the names of the files to
            read the projectors from are modified, see ``load_machine`` and
            ``load_cascade_of_machines`` methods of this class for more details.
        """

        self.load_clf_and_mean_std(projector_file)



    #==========================================================================
    def project(self, feature):
        """
        This function computes a vector of scores for each sample in the input
        array of features. The following steps are apllied:

        1. First, the input data is mean-std normalized using mean and std of the
           real class only.

        2. The input features are next classified using pre-trained GENERIC machine.

        Set ``performs_projection = True`` in the constructor to enable this function.
        It is assured that the :py:meth:`load_projector` was **called before** the
        ``project`` function is executed.

        **Parameters:**

        ``feature`` : FrameContainer or 2D :py:class:`numpy.ndarray`
            Two types of inputs are accepted.
            A Frame Container conteining the features of an individual,
            see ``bob.bio.video.utils.FrameContainer``.
            Or a 2D feature array of the size (N_samples x N_features).

        **Returns:**

        ``scores`` : 1D :py:class:`numpy.ndarray`
            Vector of scores. Scores for the real class are expected to be
            higher, than the scores of the negative / attack class.
            In this case scores are probabilities.
        """

        # 1. Convert input array to numpy array if necessary.
        if isinstance(feature, FrameContainer):  # if FrameContainer convert to 2D numpy array

            features_array = convert_frame_cont_to_array(feature)

        else:

            features_array = feature.copy()

        features_array_norm = self._normalize(features_array, train =False)

        # print(self.clf, dir(self.clf))
        if self.one_class:
            scores= self.clf.score_samples(features_array_norm)# this is available in new version -self.clf.mahalanobis(features_array_norm)  #
        else:
            scores = self.clf.predict_proba(features_array_norm)[:, 1]
        return scores

    #==========================================================================
    def score(self, toscore):
        """
        Returns a probability of a sample being a real class.

        **Parameters:**

        ``toscore`` : 1D :py:class:`numpy.ndarray`
            Vector with scores for each frame/sample defining the probability
            of the frame being a sample of the real class.

        **Returns:**

        ``score`` : [:py:class:`float`]
            If ``frame_level_scores_flag = False`` a single score is returned.
            One score per video. This score is placed into a list, because
            the ``score`` must be an iterable.
            Score is a probability of a sample being a real class.
            If ``frame_level_scores_flag = True`` a list of scores is returned.
            One score per frame/sample.
        """

        if self.frame_level_scores_flag:

            score = list(toscore)

        else:

            score = [np.mean(toscore)]  # compute a single score per video

        return score
