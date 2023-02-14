import glob
import os

import numpy as np

from click.testing import CliRunner
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import bob.measure
import bob.pipelines as mario

from bob.io.base.testing_utils import assert_click_runner_result
from bob.pad.base.error_utils import split_csv_pad
from bob.pad.base.pipelines import Database
from bob.pad.base.script.run_pipeline import run_pipeline as run_pipeline_cli


def dataset_fixed_cov():
    # dummy data from sklearn
    # code from https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html
    """Generate 2 Gaussians samples with the same covariance matrix"""
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0.0, -0.23], [0.83, 0.23]])
    X = np.r_[
        np.dot(np.random.randn(n, dim), C),
        np.dot(np.random.randn(n, dim), C) + np.array([1, 1]),
    ]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


class DummyPadDatabase(Database):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        X, y = dataset_fixed_cov()
        # split train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        self._fit_samples = [
            mario.Sample(x_, is_bonafide=bool(y_), key=str(i))
            for i, (x_, y_) in enumerate(zip(X_train, y_train))
        ]
        self._test_samples = [
            mario.Sample(x_, is_bonafide=bool(y_), key=str(i))
            for i, (x_, y_) in enumerate(zip(X_test, y_test))
        ]
        self._fit_samples = tuple(self._fit_samples)
        self._test_samples = tuple(self._test_samples)

    def fit_samples(self):
        return self._fit_samples

    def predict_samples(self, group="dev"):
        return self._test_samples


def dummy_pipeline():
    classifier = mario.wrap(
        ["sample"],
        LinearDiscriminantAnalysis(),
        fit_extra_arguments=[("y", "is_bonafide")],
    )
    pipeline = Pipeline([("lda", classifier)])
    return pipeline


def _create_config_file(path):
    with open(path, "w") as f:
        f.write(
            """
from tests.test_pipelines import DummyPadDatabase, dummy_pipeline
database = DummyPadDatabase()
pipeline = dummy_pipeline()
"""
        )


def test_run_pipeline():
    for options in [
        ["--no-dask", "--memory"],
        ["--no-dask"],
        ["--memory"],
        [],
    ]:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _create_config_file("config.py")
            result = runner.invoke(
                run_pipeline_cli,
                [
                    "-vv",
                    "config.py",
                ]
                + options,
            )
            assert_click_runner_result(result)
            generated_files = glob.glob("results/**", recursive=True)

            # check if score file is generated
            assert os.path.isfile("results/scores-dev.csv"), generated_files
            assert os.path.isfile("results/scores-eval.csv"), generated_files

            # check if model is checkpointed if not --memory
            if "--memory" not in options:
                assert os.path.isfile("results/lda.pkl"), generated_files
            else:
                assert not os.path.isfile("results/lda.pkl"), generated_files

            # test if eer is accepable
            neg, pos = split_csv_pad("results/scores-dev.csv")
            eer = bob.measure.eer(neg, pos)
            assert eer < 0.08
