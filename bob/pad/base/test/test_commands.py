from click.testing import CliRunner
import pkg_resources
from ..script import (pad_commands, vuln_commands)


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
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_det_vuln():
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
        result = runner.invoke(vuln_commands.det, ['-fnmr', '0.2',
                                                   '-o',
                                                   'DET.pdf',
                                                   licit_dev, licit_test,
                                                   spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_fmr_iapmr_vuln():
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
        result = runner.invoke(vuln_commands.fmr_iapmr, [
            '--output', 'FMRIAPMR.pdf', licit_dev, licit_test, spoof_dev,
            spoof_test
        ])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(vuln_commands.fmr_iapmr, [
            '--output', 'FMRIAPMR.pdf', licit_dev, licit_test, spoof_dev,
            spoof_test, '-G', '-L', '1e-7,1,0,1'
        ])
        assert result.exit_code == 0, (result.exit_code, result.output)


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
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.hist, ['--criterion', 'min-hter',
                                                   '--output',
                                                   'HISTO.pdf', '-b',
                                                   '30,20',
                                                   licit_dev, spoof_dev])
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(pad_commands.hist, ['-e', '--criterion', 'eer',
                                                   '--output',
                                                   'HISTO.pdf', '-b', '30',
                                                   licit_dev, licit_test,
                                                   spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_hist_vuln():
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
        result = runner.invoke(vuln_commands.hist,
                               ['--criterion', 'eer', '--output',
                                'HISTO.pdf', '-b', '30', '-ts', 'A,B',
                                licit_dev, licit_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.hist,
                               ['--criterion', 'eer', '--output',
                                'HISTO.pdf', '-b', '2,20,30', '-e',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)


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
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_epc_vuln():
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
        result = runner.invoke(vuln_commands.epc,
                               ['--output', 'epc.pdf',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(vuln_commands.epc,
                               ['--output', 'epc.pdf', '-I',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_epsc_vuln():
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
        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output, result.exception)

        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf', '-I',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf', '-D',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf', '-D',
                                '-I', '--no-wer',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert result.exit_code == 0, (result.exit_code, result.output)


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
        assert result.exit_code == 0, (result.exit_code, result.output)
