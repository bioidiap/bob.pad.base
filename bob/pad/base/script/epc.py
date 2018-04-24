"""Generates PAD ISO compliant EPC based on the score files
"""
import click
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option
from bob.bio.base.score import load
from . import figure

@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.titles_option()
@common_options.title_option()
@common_options.const_layout_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.bool_option(
    'iapmr', 'I', 'Whether to plot the IAPMR related lines or not.', True
)
@common_options.style_option()
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kwargs):
    """Plot EPC (expected performance curve):

    You need to provide 4 score
    files for each biometric system in this order:

    \b
    * licit development scores
    * licit evaluation scores
    * spoof development scores
    * spoof evaluation scores

    See :ref:`bob.pad.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    Examples:
        $ bob pad epc dev-scores eval-scores

        $ bob pad epc -o my_epc.pdf dev-scores1 eval-scores1

        $ bob pad epc {licit,spoof}/scores-{dev,eval}
    """
    process = figure.Epc(ctx, scores, True, load.split)
    process.run()

@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epsc.pdf')
@common_options.titles_option()
@common_options.const_layout_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.bool_option(
    'wer', 'w', 'Whether to plot the WER related lines or not.', True
)
@common_options.bool_option(
    'three-d', 'D', 'If true, generate 3D plots', False
)
@common_options.bool_option(
    'iapmr', 'I', 'Whether to plot the IAPMR related lines or not.', False
)
@click.option('-c', '--criteria', default="eer", show_default=True,
              help='Criteria for threshold selection',
              type=click.Choice(('eer', 'min-hter', 'bpcer20')))
@click.option('-vp', '--var-param', default="omega", show_default=True,
              help='Name of the varying parameter',
              type=click.Choice(('omega', 'beta')))
@click.option('-fp', '--fixed-param', default=0.5, show_default=True,
              help='Value of the fixed parameter',
              type=click.FLOAT)
@verbosity_option()
@click.pass_context
def epsc(ctx, scores, criteria, var_param, fixed_param, three_d, **kwargs):
    """Plot EPSC (expected performance spoofing curve):

    You need to provide 4 score
    files for each biometric system in this order:

    \b
    * licit development scores
    * licit evaluation scores
    * spoof development scores
    * spoof evaluation scores

    See :ref:`bob.pad.base.vulnerability` in the documentation for a guide on
    vulnerability analysis.

    Note that when using 3D plots with option ``--three-d``, you cannot plot
    both WER and IAPMR on the same figure (which is possible in 2D).

    Examples:
        $ bob pad epsc dev-scores eval-scores

        $ bob pad epsc -o my_epsc.pdf dev-scores1 eval-scores1

        $ bob pad epsc -D {licit,spoof}/scores-{dev,eval}
    """
    if three_d:
        if (ctx.meta['wer'] and ctx.meta['iapmr']):
            raise click.BadParameter('Cannot plot both WER and IAPMR in 3D')
        process = figure.Epsc3D(
            ctx, scores, True, load.split,
            criteria, var_param, fixed_param
        )
    else:
        process = figure.Epsc(
            ctx, scores, True, load.split,
            criteria, var_param, fixed_param
        )
    process.run()
