"""The main entry for bob.pad and its(click-based) scripts.
"""

import os
import logging
import numpy
import click
import pkg_resources
from click_plugins import with_plugins
from click.types import FLOAT
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import (verbosity_option,
                                                open_file_mode_option,
                                               bool_option)
from bob.core import random
from bob.io.base import create_directories_safe
from bob.bio.base.score import load
from . import figure

NUM_GENUINE_ACCESS = 5000
NUM_ZEIMPOSTORS = 5000
NUM_PA = 5000



@with_plugins(pkg_resources.iter_entry_points('bob.vuln.cli'))
@click.group()
def vuln():
  """Presentation Vulnerability related commands."""
  pass



def gen_score_distr(mean_gen, mean_zei, mean_pa, sigma_gen=1, sigma_zei=1,
                    sigma_pa=1):
  mt = random.mt19937()  # initialise the random number generator

  genuine_generator = random.normal(numpy.float32, mean_gen, sigma_gen)
  zei_generator = random.normal(numpy.float32, mean_zei, sigma_zei)
  pa_generator = random.normal(numpy.float32, mean_pa, sigma_pa)

  genuine_scores = [genuine_generator(mt) for i in range(NUM_GENUINE_ACCESS)]
  zei_scores = [zei_generator(mt) for i in range(NUM_ZEIMPOSTORS)]
  pa_scores = [pa_generator(mt) for i in range(NUM_PA)]

  return genuine_scores, zei_scores, pa_scores



def write_scores_to_file(neg, pos, filename, attack=False):
  """Writes score distributions into 4-column score files. For the format of
    the 4-column score files, please refer to Bob's documentation.

  Parameters
  ----------
  neg : array_like
      Scores for negative samples.
  pos : array_like
      Scores for positive samples.
  filename : str
      The path to write the score to.
  """
  create_directories_safe(os.path.dirname(filename))
  with open(filename, 'wt') as f:
      for i in pos:
          f.write('x x foo %f\n' % i)
      for i in neg:
          if attack:
              f.write('x attack foo %f\n' % i)
          else:
              f.write('x y foo %f\n' % i)



@click.command()
@click.argument('outdir')
@click.option('--mean-gen', default=10, type=FLOAT, show_default=True)
@click.option('--mean-zei', default=0, type=FLOAT, show_default=True)
@click.option('--mean-pa', default=5, type=FLOAT, show_default=True)
@verbosity_option()
def gen(outdir, mean_gen, mean_zei, mean_pa):
  """Generate random scores.
  Generates random scores for three types of verification attempts:
  genuine users, zero-effort impostors and spoofing attacks and writes them
  into 4-column score files for so called licit and spoof scenario. The
  scores are generated using Gaussian distribution whose mean is an input
  parameter. The generated scores can be used as hypothetical datasets.
  """
  # Generate the data
  genuine_dev, zei_dev, pa_dev = gen_score_distr(
      mean_gen, mean_zei, mean_pa)
  genuine_eval, zei_eval, pa_eval = gen_score_distr(
      mean_gen, mean_zei, mean_pa)

  # Write the data into files
  write_scores_to_file(genuine_dev, zei_dev,
                       os.path.join(outdir, 'licit', 'scores-dev'))
  write_scores_to_file(genuine_eval, zei_eval,
                       os.path.join(outdir, 'licit', 'scores-eval'))
  write_scores_to_file(genuine_dev, pa_dev,
                       os.path.join(outdir, 'spoof', 'scores-dev'),
                       attack=True)
  write_scores_to_file(genuine_eval, pa_eval,
                       os.path.join(outdir, 'spoof', 'scores-eval'),
                       attack=True)



@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='vuln_det.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option(dflt='upper-right')
@common_options.title_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.figsize_option()
@verbosity_option()
@common_options.axes_val_option(dflt='0.01,95,0.01,95')
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
def det(ctx, scores, criteria, real_data, **kwargs):
  """Plot DET

  You need to provide 4 scores
  files for each PAD system in this order:

  \b
  * licit development scores
  * licit evaluation scores
  * spoof development scores
  * spoof evaluation scores

  Examples:
      $ bob pad det --no-spoof dev-scores eval-scores

      $ bob pad det {licit,spoof}/scores-{dev,eval}
  """
  process = figure.Det(ctx, scores, True, load.split, criteria, real_data,
                       False)
  process.run()



@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='vuln_epc.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option()
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
@common_options.output_plot_file_option(default_out='vuln_epsc.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option()
@common_options.const_layout_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.figsize_option(dflt=None)
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



@click.command()
@common_options.scores_argument(nargs=-1, min_arg=2)
@common_options.title_option()
@common_options.output_plot_file_option(default_out='vuln_hist.pdf')
@common_options.eval_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.thresholds_option()
@common_options.print_filenames_option(dflt=False)
@bool_option(
    'iapmr-line', 'I', 'Whether to plot the IAPMR related lines or not.', True
)
@bool_option(
    'real-data', 'R',
    'If False, will annotate the plots hypothetically, instead '
    'of with real data values of the calculated error rates.', True
)
@common_options.legends_option()
@common_options.const_layout_option()
@common_options.figsize_option(dflt=None)
@common_options.subplot_option()
@common_options.legend_ncols_option()
@common_options.style_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, evaluation, **kwargs):
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

  You need to provide one or more development score file(s) for each
  experiment. You can also provide eval files along with dev files. If only
  dev-scores are used set the flag `--no-evaluation` is required in that
  case.

  By default, when eval-scores are given, only eval-scores histograms are
  displayed with threshold line
  computed from dev-scores. If you want to display dev-scores distributions
  as well, use ``--show-dev`` option.

  Examples:

      $ bob pad vuln_hist licit/scores-dev licit/scores-eval \
                          spoof/scores-dev spoof/scores-eval

      $ bob pad vuln_hist {licit,spoof}/scores-{dev,eval}
  '''
  process = figure.HistVuln(ctx, scores, evaluation, load.split)
  process.run()



@click.command(context_settings=dict(token_normalize_func=lambda x: x.lower()))
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.eval_option()
@common_options.table_option()
@common_options.criterion_option(lcriteria=['bpcer20', 'eer', 'min-hter'])
@common_options.thresholds_option()
@open_file_mode_option()
@common_options.output_log_metric_option()
@common_options.legends_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, scores, **kwargs):
  """Generate table of metrics for vulnerability PAD

  You need to provide 2 or 4 scores
  files for each PAD system in this order:

  \b
  * licit development scores
  * licit evaluation scores
  * spoof development scores
  * spoof evaluation scores


  Examples:
      $ bob pad vuln_metrics {licit,spoof}/scores-{dev,eval}
  """
  process = figure.MetricsVuln(ctx, scores, True, load.split)
  process.run()



@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.output_plot_file_option(default_out='fmr_iapmr.pdf')
@common_options.legends_option()
@common_options.no_legend_option()
@common_options.legend_loc_option()
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



@click.command()
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.table_option()
@common_options.output_log_metric_option()
@common_options.output_plot_file_option(default_out='vuln_eval.pdf')
@common_options.points_curve_option()
@common_options.lines_at_option()
@common_options.const_layout_option()
@common_options.figsize_option(dflt=None)
@common_options.style_option()
@common_options.linestyles_option()
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, **kwargs):
  '''Runs error analysis on score sets for vulnerability studies

  \b
  1. Computes bob pad vuln_metrics
  2. Plots EPC, EPSC, vulnerability histograms, fmr vs IAPMR to a multi-page
     PDF file


  You need to provide 4 score files for each biometric system in this order:

  \b
  * licit development scores
  * licit evaluation scores
  * spoof development scores
  * spoof evaluation scores

  Examples:
      $ bob pad vuln -o my_epsc.pdf dev-scores1 eval-scores1

      $ bob pad vuln -D {licit,spoof}/scores-{dev,eval}
  '''
  # first time erase if existing file
  click.echo("Computing vuln metrics...")
  ctx.invoke(metrics, scores=scores, evaluation=True)
  if 'log' in ctx.meta and ctx.meta['log'] is not None:
      click.echo("[metrics] => %s" % ctx.meta['log'])

  # avoid closing pdf file before all figures are plotted
  ctx.meta['closef'] = False
  click.echo("Computing histograms...")
  ctx.meta['criterion'] = 'eer'  # no criterion passed in evaluate
  ctx.forward(hist)  # use class defaults plot settings
  click.echo("Computing DET...")
  ctx.forward(det)  # use class defaults plot settings
  click.echo("Computing EPC...")
  ctx.forward(epc)  # use class defaults plot settings
  click.echo("Computing EPSC...")
  ctx.forward(epsc)  # use class defaults plot settings
  click.echo("Computing FMR vs IAPMR...")
  ctx.meta['closef'] = True
  ctx.forward(fmr_iapmr)  # use class defaults plot settings
  click.echo("Vuln successfully completed!")
  click.echo("[plots] => %s" % (ctx.meta['output']))
