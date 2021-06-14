from click.testing import CliRunner
import pkg_resources
from ..script import pad_commands
from bob.extension.scripts.click_helper import assert_click_runner_result
import os


def test_gen_pad():
    dev_ref_file = pkg_resources.resource_filename(
        "bob.pad.base.test", "data/csv_scores/scores-dev.csv"
    )
    eval_ref_file = pkg_resources.resource_filename(
        "bob.pad.base.test", "data/csv_scores/scores-eval.csv"
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
    licit_dev = pkg_resources.resource_filename('bob.pad.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.pad.base.test',
                                                 'data/licit/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.det, ['-e', '--output',
                                                  'DET.pdf',
                                                  licit_dev, licit_test])
        assert_click_runner_result(result)


def test_hist_pad():
    licit_dev = pkg_resources.resource_filename('bob.pad.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.pad.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.pad.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.pad.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.hist, [licit_dev])
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.hist, ['--criterion', 'min-hter',
                                                   '--output',
                                                   'HISTO.pdf', '-b',
                                                   '30,20',
                                                   licit_dev, spoof_dev])
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.hist, ['-e', '--criterion', 'eer',
                                                   '--output',
                                                   'HISTO.pdf', '-b', '30',
                                                   licit_dev, licit_test,
                                                   spoof_dev, spoof_test])
        assert_click_runner_result(result)


def test_metrics_pad():
    licit_dev = pkg_resources.resource_filename('bob.pad.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.pad.base.test',
                                                 'data/licit/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            pad_commands.metrics,
            ['-e', licit_dev, licit_test]
        )
        assert_click_runner_result(result)


def test_evaluate_pad():
    licit_dev = pkg_resources.resource_filename('bob.pad.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.pad.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.pad.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.pad.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.evaluate,
                               [licit_dev, licit_test, spoof_dev, spoof_test])
        assert_click_runner_result(result)
