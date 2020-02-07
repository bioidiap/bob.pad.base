#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 19 Oct 23:43:22 2016


import logging
import numpy

import bob.io.base
import bob.learn.linear
import bob.learn.em
from bob.bio.video import FrameContainer
from . import Algorithm
from ..utils import (
    convert_and_prepare_features,
)


logger = logging.getLogger(__name__)


class GMM(Algorithm):
    """Trains two GMMs for two classes of PAD and calculates log likelihood ratio during
    evaluation.
    """

    def __init__(
        self,
        # parameters for the GMM
        number_of_gaussians,
        # parameters of UBM training
        kmeans_training_iterations=25,  # Maximum number of iterations for K-Means
        gmm_training_iterations=10,  # Maximum number of iterations for ML GMM Training
        training_threshold=5e-4,  # Threshold to end the ML training
        variance_threshold=5e-4,  # Minimum value that a variance can reach
        update_weights=True,
        update_means=True,
        update_variances=True,
        responsibility_threshold=0,
        # If set, the weight of a particular Gaussian will at least be greater than this
        # threshold. In the case the real weight is lower, the prior mean value will be
        # used to estimate the current mean and variance.
        INIT_SEED=5489,
        performs_projection=True,
        requires_projector_training=True,
        **kwargs,
    ):

        super().__init__(
            performs_projection=performs_projection,
            requires_projector_training=requires_projector_training,
            **kwargs,
        )
        self._kwargs.update(
            dict(
                number_of_gaussians=number_of_gaussians,
                kmeans_training_iterations=kmeans_training_iterations,
                gmm_training_iterations=gmm_training_iterations,
                training_threshold=training_threshold,
                variance_threshold=variance_threshold,
                update_weights=update_weights,
                update_means=update_means,
                update_variances=update_variances,
                responsibility_threshold=responsibility_threshold,
                INIT_SEED=INIT_SEED,
            )
        )

        # copy parameters
        self.gaussians = number_of_gaussians
        self.kmeans_training_iterations = kmeans_training_iterations
        self.gmm_training_iterations = gmm_training_iterations
        self.training_threshold = training_threshold
        self.variance_threshold = variance_threshold
        self.update_weights = update_weights
        self.update_means = update_means
        self.update_variances = update_variances
        self.responsibility_threshold = responsibility_threshold
        self.init_seed = INIT_SEED
        self.rng = bob.core.random.mt19937(self.init_seed)

        self.gmm_machine_real = None
        self.gmm_machine_attack = None
        self.kmeans_trainer = bob.learn.em.KMeansTrainer()
        self.gmm_trainer = bob.learn.em.ML_GMMTrainer(
            self.update_means,
            self.update_variances,
            self.update_weights,
            self.responsibility_threshold,
        )

    def _check_feature(self, feature, machine=None, projected=False):
        """Checks that the features are appropriate."""
        if (
            not isinstance(feature, numpy.ndarray)
            or feature.ndim != 2
            or feature.dtype != numpy.float64
        ):
            raise ValueError("The given feature is not appropriate", feature)
        if (
            self.gmm_machine_real is not None
            and feature.shape[1] != self.gmm_machine_real.shape[1]
        ):
            raise ValueError(
                "The given feature is expected to have %d elements, but it has %d"
                % (self.gmm_machine_real.shape[1], feature.shape[1])
            )
        if (
            self.gmm_machine_attack is not None
            and feature.shape[1] != self.gmm_machine_attack.shape[1]
        ):
            raise ValueError(
                "The given feature is expected to have %d elements, but it has %d"
                % (self.gmm_machine_attack.shape[1], feature.shape[1])
            )
        return True

    #######################################################
    #                  GMM training                       #

    def train_gmm(self, array):

        logger.debug(" .... Training with %d feature vectors", array.shape[0])

        # Computes input size
        input_size = array.shape[1]

        # Creates the machines (KMeans and GMM)
        logger.debug(" .... Creating machines")
        kmeans_machine = bob.learn.em.KMeansMachine(self.gaussians, input_size)
        gmm_machine = bob.learn.em.GMMMachine(self.gaussians, input_size)

        # initialize the random generator with out one single cool seed that allows us to reproduce experiments
        logger.info("  -> Init random generator with seed %d", self.init_seed)
        self.rng = bob.core.random.mt19937(self.init_seed)

        # Trains using the KMeansTrainer
        logger.info("  -> Training K-Means")
        bob.learn.em.train(
            self.kmeans_trainer,
            kmeans_machine,
            array,
            self.kmeans_training_iterations,
            self.training_threshold,
            self.rng,
        )

        variances, weights = kmeans_machine.get_variances_and_weights_for_each_cluster(
            array
        )
        means = kmeans_machine.means

        # Initializes the GMM
        gmm_machine.means = means
        gmm_machine.variances = variances
        gmm_machine.weights = weights
        gmm_machine.set_variance_thresholds(self.variance_threshold)

        # Trains the GMM
        logger.info("  -> Training GMM")
        bob.learn.em.train(
            self.gmm_trainer,
            gmm_machine,
            array,
            self.gmm_training_iterations,
            self.training_threshold,
            self.rng,
        )
        return gmm_machine

    def save_gmms(self, projector_file):
        """Save projector to file"""
        # Saves the trained GMMs to file
        logger.debug(" .... Saving GMM models to file '%s'", projector_file)
        hdf5 = (
            projector_file
            if isinstance(projector_file, bob.io.base.HDF5File)
            else bob.io.base.HDF5File(projector_file, "w")
        )
        hdf5.create_group("GMMReal")
        hdf5.cd("GMMReal")
        self.gmm_machine_real.save(hdf5)
        hdf5.cd("/")
        hdf5.create_group("GMMAttack")
        hdf5.cd("GMMAttack")
        self.gmm_machine_attack.save(hdf5)

    def train_projector(self, training_features, projector_file):
        if len(training_features) != 2:
            raise ValueError(
                "Training projector: features should contain two lists: real and attack!"
            )

        logger.info(
            " - Training: number of real features %d", len(training_features[0])
        )
        logger.info(
            " - Training: number of attack features %d", len(training_features[1])
        )

        attack_features = convert_and_prepare_features(
            training_features[1], dtype="float64"
        )
        del training_features[1]
        real_features = convert_and_prepare_features(
            training_features[0], dtype="float64"
        )
        del training_features

        [self._check_feature(feature) for feature in real_features]
        [self._check_feature(feature) for feature in attack_features]

        logger.debug(
            "GMM:train_projector(), real_features shape: %s", real_features.shape
        )
        logger.debug(
            "GMM:train_projector(), attack_features shape: %s", attack_features.shape
        )
        logger.debug("Min real %g", numpy.min(real_features))
        logger.debug("Max real %g", numpy.max(real_features))
        logger.debug("Min attack %g", numpy.min(attack_features))
        logger.debug("Max attack %g", numpy.max(attack_features))

        logger.info("Training the GMM for real samples")
        self.gmm_machine_real = self.train_gmm(real_features)
        logger.info("Training the GMM for attack samples")
        self.gmm_machine_attack = self.train_gmm(attack_features)

        logger.info("Saving the GMMs")
        self.save_gmms(projector_file)

    def load_projector(self, projector_file):

        with bob.io.base.HDF5File(projector_file) as hdf5file:
            # read GMM for real data
            hdf5file.cd("/GMMReal")
            self.gmm_machine_real = bob.learn.em.GMMMachine(hdf5file)
            # read GMM for attack data
            hdf5file.cd("/GMMAttack")
            self.gmm_machine_attack = bob.learn.em.GMMMachine(hdf5file)

        self.gmm_machine_real.set_variance_thresholds(self.variance_threshold)
        self.gmm_machine_attack.set_variance_thresholds(self.variance_threshold)

    def project(self, feature):
        """project(feature) -> projected

        Projects the given feature into GMM space.

        **Parameters:**

        feature : 1D :py:class:`numpy.ndarray`
          The 1D feature to be projected.

        **Returns:**

        projected : 1D :py:class:`numpy.ndarray`
          The ``feature`` projected into GMM space.
        """
        if isinstance(feature, FrameContainer):
            feature = feature.as_array()

        feature = numpy.asarray(feature, dtype=numpy.float64)
        self._check_feature(feature)

        logger.debug(" .... Projecting %d features vector" % feature.shape[0])

        # return the resulting log likelihoods
        return numpy.asarray(
            [self.gmm_machine_real(feature), self.gmm_machine_attack(feature)],
            dtype=numpy.float64,
        )

    def score(self, toscore):
        """Returns the difference between log likelihoods of being real or attack"""
        return [toscore[0] - toscore[1]]

    def score_for_multiple_projections(self, toscore):
        """Returns the difference between log likelihoods of being real or attack"""
        return self.score(toscore)
