"""Generates PAD ISO compliant EPC based on the score files
"""
import click
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option
from bob.bio.base.score import load
from . import figure

@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.legends_option()
@common_options.title_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option()
@common_options.bool_option(
    'no-spoof', 'ns', '', False
)
@verbosity_option()
@common_options.axes_val_option(dflt=[0.01, 95, 0.01, 95])
@common_options.x_rotation_option(dflt=45)
@common_options.x_label_option()
@common_options.y_label_option()
@click.option('-c', '--criteria', default=None, show_default=True,
              help='Criteria for threshold selection',
              type=click.Choice(('eer', 'min-hter', 'bpcer20')))
@click.option('--real-data/--no-real-data', default=True, show_default=True,
              help='If False, will annotate the plots hypothetically, instead '
              'of with real data values of the calculated error rates.')
@click.pass_context
def det(ctx, scores, criteria,  real_data, **kwargs):
    """Plot DET

    You need to provide 2 or 4 scores
    files for each PAD system in this order:

    \b
    * licit development scores
    * licit evaluation scores
    * spoof development scores (when ``--no-spoof`` is False (default))
    * spoof evaluation scores (when ``--no-spoof`` is False (default))


    Examples:
        $ bob pad det --no-spoof dev-scores eval-scores

        $ bob pad det {licit,spoof}/scores-{dev,eval}
    """
    process = figure.Det(ctx, scores, True, load.split, criteria, real_data)
    process.run()
