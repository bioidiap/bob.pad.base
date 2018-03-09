from bob.pad.base.algorithm import Algorithm


class Predictions(Algorithm):
    """An algorithm that takes the precomputed predictions and uses them for
    scoring."""

    def __init__(self, **kwargs):
        super(Predictions, self).__init__(
            **kwargs)

    def score(self, predictions):
        # Assuming the predictions are the output of a softmax layer
        return [predictions[1]]
