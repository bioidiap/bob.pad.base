"""Generates PAD ISO compliant FMR vs IAPMR plots based on the score files
"""
import click
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option
from bob.bio.base.score import load
from . import figure

@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='fmr_iapmr.pdf')
@common_options.legends_option()
@common_options.title_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option()
@verbosity_option()
@common_options.axes_val_option()
@common_options.x_rotation_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.semilogx_option()
@click.pass_context
def fmr_iapmr(ctx, scores, **kwargs):
    """Plot FMR vs IAPMR

    You need to provide 2 or 4 scores
    files for each PAD system in this order:

    \b
    * licit development scores
    * licit evaluation scores
    * spoof development scores (when ``--no-spoof`` is False (default))
    * spoof evaluation scores (when ``--no-spoof`` is False (default))


    Examples:
        $ bob pad fmr_iapmr --no-spoof dev-scores eval-scores

        $ bob pad fmr_iapmr {licit,spoof}/scores-{dev,eval}
    """
    process = figure.FmrIapmr(ctx, scores, True, load.split)
    process.run()
