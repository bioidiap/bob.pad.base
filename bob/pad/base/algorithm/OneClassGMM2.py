# -*- coding: utf-8 -*-
# @author: Amir Mohammadi

from bob.pad.base.algorithm import Algorithm
from bob.pad.base.utils import convert_and_prepare_features
from bob.bio.gmm.algorithm import GMM
import logging
import numpy as np
from collections.abc import Iterable
from multiprocessing import cpu_count
from bob.bio.video.utils import FrameContainer

from bob.pad.base.utils import convert_frame_cont_to_array, mean_std_normalize, convert_and_prepare_features

logger = logging.getLogger(__name__)


def bic(trainer, machine, X):
    """Bayesian information criterion for the current model on the input X.

    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)

    Returns
    -------
    bic : float
        The lower the better.
    """
    log_likelihood = trainer.compute_likelihood(machine)
    n_parameters = (
        machine.means.size + machine.variances.size + len(machine.weights) - 1
    )
    return -2 * log_likelihood * X.shape[0] + n_parameters * np.log(X.shape[0])


class OneClassGMM2(Algorithm):
    """A one class GMM implementation based on Bob's GMM implementation which is more
    stable than scikit-learn's one."""

    def __init__(
        self,
        # parameters for the GMM
        number_of_gaussians,
        # parameters of UBM training
        kmeans_training_iterations=25,  # Maximum number of iterations for K-Means
        gmm_training_iterations=25,  # Maximum number of iterations for ML GMM Training
        training_threshold=5e-4,  # Threshold to end the ML training
        variance_threshold=5e-4,  # Minimum value that a variance can reach
        update_weights=True,
        update_means=True,
        update_variances=True,
        n_threads=4,
        frame_level_scores_flag=True,
        **kwargs
    ):
        kwargs.setdefault("performs_projection", True)
        kwargs.setdefault("requires_projector_training", True)
        super().__init__(**kwargs)
        self.gmm_alg = GMM(
            number_of_gaussians=number_of_gaussians,
            kmeans_training_iterations=kmeans_training_iterations,
            gmm_training_iterations=gmm_training_iterations,
            training_threshold=training_threshold,
            variance_threshold=variance_threshold,
            update_weights=update_weights,
            update_means=update_means,
            update_variances=update_variances,
        )
        self.number_of_gaussians = number_of_gaussians
        self.frame_level_scores_flag =frame_level_scores_flag

    def train_projector(self, training_features, projector_file):
        del training_features[1]
        real = convert_and_prepare_features(training_features[0], dtype="float64")
        del training_features[0]

        if isinstance(self.number_of_gaussians, Iterable):
            logger.info(
                "Performing grid search for GMM on number_of_gaussians: %s",
                self.number_of_gaussians,
            )
            lowest_bic = np.infty
            best_n_gaussians = None
            for nc in self.number_of_gaussians:
                logger.info("Testing for number_of_gaussians: %s", nc)
                self.gmm_alg.gaussians = nc
                self.gmm_alg.train_ubm(real)
                bic_ = bic(self.gmm_alg.ubm_trainer, self.gmm_alg.ubm, real)
                logger.info("BIC for number_of_gaussians: %s is %s", nc, bic_)
                if bic_ < lowest_bic:
                    gmm = self.gmm_alg.ubm
                    lowest_bic = bic_
                    best_n_gaussians = nc
                    logger.info("Best parameters so far: number_of_gaussians %s", nc)

            assert best_n_gaussians is not None
            self.gmm_alg.gaussians = best_n_gaussians
        else:
            self.gmm_alg.train_ubm(real)
            gmm = self.gmm_alg.ubm

        self.gmm_alg.ubm = gmm
        self.gmm_alg.save_ubm(projector_file)

    def load_projector(self, projector_file):
        self.gmm_alg.load_ubm(projector_file)

    def project(self, feature):


        if isinstance(
                feature,
                FrameContainer):  # if FrameContainer convert to 2D numpy array

            features_array = convert_frame_cont_to_array(feature)

        else:

            features_array = feature


        print('features_array',features_array.shape)
        
        scores=[]


        for feat in features_array:

            score = self.gmm_alg.ubm(feat)
            scores.append(score)

        return np.array(scores)

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

        print('toscore',toscore.shape)
        if self.frame_level_scores_flag:

            score = list(toscore)

        else:

            score = [np.mean(toscore)]  # compute a single score per video

        return score
