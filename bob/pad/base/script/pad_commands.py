"""The main entry for bob.pad and its (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import (verbosity_option,
                                                open_file_mode_option,
                                                AliasedGroup)
import bob.bio.base.script.gen as bio_gen
import bob.bio.base.script.figure as bio_figure
import bob.measure.script.figure as measure_figure
from bob.bio.base.score import load
from . import figure



@click.command()
@click.argument('outdir')
@click.option('-mm', '--mean-match', default=10, type=click.FLOAT, show_default=True)
@click.option('-mnm', '--mean-non-match', default=-10,
              type=click.FLOAT, show_default=True)
@click.option('-n', '--n-sys', default=1, type=click.INT, show_default=True)
@click.option('--five-col/--four-col', default=False, show_default=True)
@verbosity_option()
@click.pass_context
def gen(ctx, outdir, mean_match, mean_non_match, n_sys, five_col):
  """Generate random scores.
  Generates random scores in 4col or 5col format. The scores are generated
  using Gaussian distribution whose mean is an input
  parameter. The generated scores can be used as hypothetical datasets.
  Invokes :py:func:`bob.bio.base.script.commands.gen`.
  """
  ctx.forward(bio_gen.gen)



@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.legends_option()
@common_options.legend_loc_option(dflt='lower-right')
@common_options.no_legend_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.eval_option()
@common_options.points_curve_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt='1e-4,1,1e-4,1')
@common_options.x_rotation_option()
@common_options.lines_at_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option(dflt=None)
@common_options.min_far_option()
@verbosity_option()
@click.pass_context
def roc(ctx, scores, evaluation, **kargs):
  """Plot ROC (receiver operating characteristic) curve:
  The plot will represent the false match rate on the horizontal axis and the
  false non match rate on the vertical axis.  The values for the axis will be
  computed using :py:func:`bob.measure.roc`.

  You need to provide one or more development score file(s) for each
  experiment. You can also provide eval files along with dev files. If only
  dev-scores are used, the flag `--no-evaluation` must be used. is required
  in that case. Files must be 4-col format, see
  :py:func:`bob.bio.base.score.load.four_column`
  Examples:
      $ bob pad roc -v dev-scores

      $ bob pad roc -v dev-scores1 eval-scores1 dev-scores2
      eval-scores2

      $ bob pad roc -v -o my_roc.pdf dev-scores1 eval-scores1
  """
  process = figure.Roc(ctx, scores, evaluation, load.split)
  process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.legends_option()
@common_options.legend_loc_option(dflt='upper-right')
@common_options.no_legend_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.sep_dev_eval_option()
@common_options.eval_option()
@common_options.axes_val_option(dflt='0.01,95,0.01,95')
@common_options.x_rotation_option(dflt=45)
@common_options.points_curve_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option(dflt=None)
@common_options.lines_at_option()
@common_options.min_far_option()
@verbosity_option()
@click.pass_context
def det(ctx, scores, evaluation, **kargs):
  """Plot DET (detection error trade-off) curve:
  modified ROC curve which plots error rates on both axes
  (false positives on the x-axis and false negatives on the y-axis)

  You need to provide one or more development score file(s) for each
  experiment. You can also provide eval files along with dev files. If only
  dev-scores are used, the flag `--no-evaluation` must be used. is required
  in that case. Files must be 4-col format, see
  :py:func:`bob.bio.base.score.load.four_column` for details.

  Examples:
    $ bob pad det -v dev-scores eval-scores

    $ bob pad det -v scores-{dev,eval}
  """
  process = figure.DetPad(ctx, scores, evaluation, load.split)
  process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.eval_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.far_option()
@common_options.thresholds_option()
@common_options.const_layout_option()
@common_options.print_filenames_option(dflt=False)
@common_options.legends_option()
@common_options.figsize_option(dflt=None)
@common_options.subplot_option()
@common_options.legend_ncols_option()
@common_options.style_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, evaluation, **kwargs):
  """ Plots histograms of Bona fida and PA along with threshold
  criterion.

  You need to provide one or more development score file(s) for each
  experiment. You can also provide eval files along with dev files. If only
  dev scores are provided, you must use flag `--no-evaluation`.

  By default, when eval-scores are given, only eval-scores histograms are
  displayed with threshold line
  computed from dev-scores.

  Examples:
      $ bob pad hist -v dev-scores

      $ bob pad hist -v dev-scores1 eval-scores1 dev-scores2
      eval-scores2

      $ bob pad hist -v --criterion min-hter dev-scores1 eval-scores1
  """
  process = figure.HistPad(ctx, scores, evaluation, load.split)
  process.run()



@click.command()
@common_options.scores_argument(min_arg=1, force_eval=True, nargs=-1)
@common_options.titles_option()
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.legends_option()
@common_options.legend_loc_option(dflt='upper-center')
@common_options.no_legend_option()
@common_options.points_curve_option()
@common_options.const_layout_option()
@common_options.style_option()
@common_options.linestyles_option()
@common_options.figsize_option(dflt=None)
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kargs):
  """Plot EPC (expected performance curve):
  plots the error rate on the eval set depending on a threshold selected
  a-priori on the development set and accounts for varying relative cost
  in [0; 1] of FPR and FNR when calculating the threshold.

  You need to provide one or more development score and eval file(s)
  for each experiment. Files must be 4-columns format, see
  :py:func:`bob.bio.base.score.load.four_column` for details.

  Examples:
      $ bob pad epc -v dev-scores eval-scores

      $ bob pad epc -v -o my_epc.pdf dev-scores1 eval-scores1
  """
  process = measure_figure.Epc(ctx, scores, True, load.split)
  process.run()



@click.command(context_settings=dict(token_normalize_func=lambda x: x.lower()))
@common_options.scores_argument(nargs=-1)
@common_options.eval_option()
@common_options.table_option()
@open_file_mode_option()
@common_options.output_log_metric_option()
@common_options.legends_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, scores, evaluation, **kwargs):
  """PAD ISO compliant metrics.

  Reports several metrics based on a selected thresholds on the development
  set and apply them on evaluation sets (if provided). The used thresholds
  are:

      bpcer20     When APCER is set to 5%.

      eer         When BPCER == APCER.

      min-hter    When HTER is minimum.

  This command produces one table per sytem. Format of the table can be
  changed through option ``--tablefmt``.

  Most metrics are according to the ISO/IEC 30107-3:2017 "Information
  technology -- Biometric presentation attack detection -- Part 3: Testing
  and reporting" standard. The reported metrics are:

      APCER: Attack Presentation Classification Error Rate

      BPCER: Bona-fide Presentation Classification Error Rate

      HTER (non-ISO): Half Total Error Rate ((BPCER+APCER)/2)

  Examples:

      $ bob pad metrics /path/to/scores-dev

      $ bob pad metrics /path/to/scores-dev /path/to/scores-eval

      $ bob pad metrics /path/to/system{1,2,3}/score-{dev,eval}
  """
  process = figure.Metrics(ctx, scores, evaluation, load.split)
  process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.legends_option()
@common_options.sep_dev_eval_option()
@common_options.table_option()
@common_options.eval_option()
@common_options.output_log_metric_option()
@common_options.output_plot_file_option(default_out='eval_plots.pdf')
@common_options.points_curve_option()
@common_options.lines_at_option()
@common_options.const_layout_option()
@common_options.figsize_option(dflt=None)
@common_options.style_option()
@common_options.linestyles_option()
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, evaluation, **kwargs):
  '''Runs error analysis on score sets

  \b
  1. Computes the threshold using either EER or min. HTER criteria on
     development set scores
  2. Applies the above threshold on evaluation set scores to compute the
     HTER, if a eval-score set is provided
  3. Reports error rates on the console
  4. Plots ROC, EPC, DET curves and score distributions to a multi-page PDF
     file


  You need to provide 2 score files for each biometric system in this order:

  \b
  * development scores
  * evaluation scores

  Examples:
      $ bob pad evaluate -v dev-scores

      $ bob pad evaluate -v scores-dev1 scores-eval1 scores-dev2
      scores-eval2

      $ bob pad evaluate -v /path/to/sys-{1,2,3}/scores-{dev,eval}

      $ bob pad evaluate -v -l metrics.txt -o my_plots.pdf dev-scores eval-scores
  '''
  # first time erase if existing file
  click.echo("Computing metrics...")
  ctx.invoke(metrics, scores=scores, evaluation=evaluation)
  if 'log' in ctx.meta and ctx.meta['log'] is not None:
      click.echo("[metrics] => %s" % ctx.meta['log'])

  # avoid closing pdf file before all figures are plotted
  ctx.meta['closef'] = False
  if evaluation:
      click.echo("Starting evaluate with dev and eval scores...")
  else:
      click.echo("Starting evaluate with dev scores only...")
  click.echo("Computing ROC...")
  # set axes limits for ROC
  ctx.forward(roc)  # use class defaults plot settings
  click.echo("Computing DET...")
  ctx.forward(det)  # use class defaults plot settings
  # the last one closes the file
  ctx.meta['closef'] = True
  click.echo("Computing score histograms...")
  ctx.meta['criterion'] = 'eer'  # no criterion passed in evaluate
  ctx.forward(hist)
  click.echo("Evaluate successfully completed!")
  click.echo("[plots] => %s" % (ctx.meta['output']))
