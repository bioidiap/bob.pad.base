import os

import pkg_resources

from click.testing import CliRunner

from bob.extension.scripts.click_helper import assert_click_runner_result

from ..script import pad_commands


def test_gen_pad():
    dev_ref_file = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-dev.csv"
    )
    eval_ref_file = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-eval.csv"
    )
    with open(dev_ref_file, "r") as f:
        dev_ref = f.readlines()
    with open(eval_ref_file, "r") as f:
        eval_ref = f.readlines()
    runner = CliRunner()
    with runner.isolated_filesystem():
        cwd = "./"
        result = runner.invoke(
            pad_commands.gen,
            [
                cwd,
                "--mean-match",
                "10",
                "--mean-attacks",
                "9",
                "--mean-attacks",
                "6",
                "--n-attacks",
                "2",
            ],
        )
        assert_click_runner_result(result)
        with open(os.path.join(cwd, "scores-dev.csv"), "r") as f:
            for generated_line, reference in zip(f.readlines(), dev_ref):
                assert generated_line == reference
        with open(os.path.join(cwd, "scores-eval.csv"), "r") as f:
            for generated_line, reference in zip(f.readlines(), eval_ref):
                assert generated_line == reference


def test_det_pad():
    dev = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-dev.csv"
    )
    test = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-eval.csv"
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            pad_commands.det, ["-e", "--output", "DET.pdf", dev, test]
        )
        assert_click_runner_result(result)


def test_hist_pad():
    dev = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-dev.csv"
    )
    test = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-eval.csv"
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.hist, [dev])
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            pad_commands.hist,
            [
                "--criterion",
                "min-hter",
                "--output",
                "HISTO.pdf",
                "-b",
                "30,20",
                dev,
            ],
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            pad_commands.hist,
            [
                "-e",
                "--criterion",
                "eer",
                "--output",
                "HISTO.pdf",
                "-b",
                "30",
                dev,
                test,
            ],
        )
        assert_click_runner_result(result)


def test_metrics_pad():
    dev = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-dev.csv"
    )
    test = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-eval.csv"
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.metrics, ["-e", dev, test, "-vvv"])
        assert_click_runner_result(result)


def test_evaluate_pad():
    dev = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-dev.csv"
    )
    test = pkg_resources.resource_filename(
        __name__, "data/csv_scores/scores-eval.csv"
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.evaluate, [dev, test])
        assert_click_runner_result(result)
