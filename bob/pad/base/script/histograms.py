"""Generates PAD ISO compliant histograms based on the score files
"""
import click
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option
from bob.bio.base.score import load
from . import figure

FUNC_SPLIT = lambda x: load.load_files(x, load.split)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.eval_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.axis_fontsize_option()
@common_options.thresholds_option()
@common_options.const_layout_option()
@common_options.show_dev_option()
@common_options.print_filenames_option(dflt=False)
@common_options.titles_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, evaluation, **kwargs):
    """ Plots histograms of Bona fida and PA along with threshold
    criterion.

    You need provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev scores
    are provided, you must use flag `--no-evaluation`.

    By default, when eval-scores are given, only eval-scores histograms are 
    displayed with threshold line
    computed from dev-scores. If you want to display dev-scores distributions 
    as well, use ``--show-dev`` option.

    Examples:
        $ bob pad hist dev-scores

        $ bob pad hist dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob pad hist --criter hter dev-scores1 eval-scores1
    """
    process = figure.HistPad(ctx, scores, evaluation, FUNC_SPLIT)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1, eval_mandatory=True, min_len=2)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.eval_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.axis_fontsize_option()
@common_options.thresholds_option()
@common_options.const_layout_option()
@common_options.show_dev_option()
@common_options.print_filenames_option(dflt=False)
@common_options.bool_option(
    'iapmr-line', 'I', 'Whether to plot the IAPMR related lines or not.', True
)
@common_options.bool_option(
    'real-data', 'R',
    'If False, will annotate the plots hypothetically, instead '
    'of with real data values of the calculated error rates.', True
)
@common_options.titles_option()
@verbosity_option()
@click.pass_context
def vuln(ctx, scores, evaluation, **kwargs):
    '''Vulnerability analysis distributions.

    Plots the histogram of score distributions. You need to provide 4 score
    files for each biometric system in this order:

    \b
    * licit development scores
    * licit evaluation scores
    * spoof development scores
    * spoof evaluation scores

    See :ref:`bob.pad.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    You need provide one or more development score file(s) for each experiment.
    You can also provide eval files along with dev files. If only dev-scores
    are used set the flag `--no-evaluation`
    is required in that case.

    By default, when eval-scores are given, only eval-scores histograms are
    displayed with threshold line
    computed from dev-scores. If you want to display dev-scores distributions
    as well, use ``--show-dev`` option.

    Examples:

        $ bob pad vuln licit/scores-dev licit/scores-eval \
                            spoof/scores-dev spoof/scores-eval

        $ bob pad vuln {licit,spoof}/scores-{dev,eval}
    '''
    process = figure.HistVuln(ctx, scores, evaluation, FUNC_SPLIT)
    process.run()
