from abc import ABCMeta, abstractmethod


class Database(metaclass=ABCMeta):
    """Base database class for PAD experiments."""

    @abstractmethod
    def fit_samples(self):
        """Returns :any:`bob.pipelines.Sample`'s to train a PAD model.

        Returns
        -------
        samples : list
            List of samples for model training.
        """
        pass

    @abstractmethod
    def predict_samples(self, group="dev"):
        """Returns :any:`bob.pipelines.Sample`'s to be scored.

        Parameters
        ----------
        group : :py:class:`str`, optional
            Limits samples to this group

        Returns
        -------
        samples : list
            List of samples to be scored.
        """
        pass
