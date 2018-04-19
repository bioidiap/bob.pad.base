import sys
import click
from click.testing import CliRunner
import pkg_resources
from ..script import (metrics, histograms, epc, det)


def test_det():
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
        result = runner.invoke(det.det, ['-c', 'min-hter', 
                                                 '--output',
                                                 'DET.pdf',
                                                 licit_dev, licit_test,
                                                 spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_hist():
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
        result = runner.invoke(histograms.hist, ['--no-evaluation', licit_dev])
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(histograms.hist, ['--criter', 'hter', '--output',
                                                 'HISTO.pdf', '-b',
                                                 30, '--no-evaluation',
                                                 licit_dev, spoof_dev])
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(histograms.hist, ['--criter', 'eer', '--output',
                                                 'HISTO.pdf', '-b', 30,
                                                 licit_dev, licit_test,
                                                 spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_vuln():
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
        result = runner.invoke(histograms.vuln, ['--criter', 'eer', '--output',
                                                 'HISTO.pdf', '-b', 30,
                                                 licit_dev, licit_test,
                                                 spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_epc():
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
        result = runner.invoke(epc.epc, ['--output', 'epc.pdf',
                                         licit_dev, licit_test,
                                         spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(epc.epc, ['--output', 'epc.pdf', '-I',
                                         licit_dev, licit_test,
                                         spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_epsc():
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
        result = runner.invoke(epc.epsc, ['--output', 'epsc.pdf',
                                          licit_dev, licit_test,
                                          spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(epc.epsc, ['--output', 'epsc.pdf', '-I',
                                          licit_dev, licit_test,
                                          spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(epc.epsc, ['--output', 'epsc.pdf', '-D',
                                          licit_dev, licit_test,
                                          spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(epc.epsc, ['--output', 'epsc.pdf', '-D',
                                          '-I', '--no-wer',
                                          licit_dev, licit_test,
                                          spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)
