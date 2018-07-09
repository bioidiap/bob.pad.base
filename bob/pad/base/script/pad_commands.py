"""The main entry for bob pad commands.
"""
import click
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import verbosity_option
import bob.bio.base.script.gen as bio_gen
import bob.measure.script.figure as measure_figure
from bob.bio.base.score import load
from . import pad_figure as figure

SCORE_FORMAT = (
    "Files must be 4-col format, see "
    ":py:func:`bob.bio.base.score.load.four_column`.")
CRITERIA = ('eer', 'min-hter', 'bpcer20')


@click.command()
@click.argument('outdir')
@click.option('-mm', '--mean-match', default=10, type=click.FLOAT,
              show_default=True)
@click.option('-mnm', '--mean-non-match', default=-10,
              type=click.FLOAT, show_default=True)
@click.option('-n', '--n-sys', default=1, type=click.INT, show_default=True)
@verbosity_option()
@click.pass_context
def gen(ctx, outdir, mean_match, mean_non_match, n_sys, **kwargs):
  """Generate random scores.
  Generates random scores in 4col or 5col format. The scores are generated
  using Gaussian distribution whose mean is an input
  parameter. The generated scores can be used as hypothetical datasets.
  Invokes :py:func:`bob.bio.base.script.commands.gen`.
  """
  ctx.meta['five_col'] = False
  ctx.forward(bio_gen.gen)


@common_options.metrics_command(common_options.METRICS_HELP.format(
    names='FtA, APCER, BPCER, FAR, FRR, ACER',
    criteria=CRITERIA, score_format=SCORE_FORMAT,
    hter_note='Note that FAR = APCER * (1 - FtA), '
    'FRR = FtA + BPCER * (1 - FtA) and ACER = (APCER + BPCER) / 2.',
    command='bob pad metrics'), criteria=CRITERIA)
def metrics(ctx, scores, evaluation, **kwargs):
  process = figure.Metrics(ctx, scores, evaluation, load.split)
  process.run()


@common_options.roc_command(
    common_options.ROC_HELP.format(
        score_format=SCORE_FORMAT, command='bob pad roc'))
def roc(ctx, scores, evaluation, **kwargs):
  process = figure.Roc(ctx, scores, evaluation, load.split)
  process.run()


@common_options.det_command(
    common_options.DET_HELP.format(
        score_format=SCORE_FORMAT, command='bob pad det'))
def det(ctx, scores, evaluation, **kwargs):
  process = figure.Det(ctx, scores, evaluation, load.split)
  process.run()


@common_options.epc_command(
    common_options.EPC_HELP.format(
        score_format=SCORE_FORMAT, command='bob pad epc'))
def epc(ctx, scores, **kwargs):
  process = measure_figure.Epc(ctx, scores, True, load.split, hter='ACER')
  process.run()


@common_options.hist_command(
    common_options.HIST_HELP.format(
        score_format=SCORE_FORMAT, command='bob pad hist'))
def hist(ctx, scores, evaluation, **kwargs):
  process = figure.Hist(ctx, scores, evaluation, load.split)
  process.run()


@common_options.evaluate_command(
    common_options.EVALUATE_HELP.format(
        score_format=SCORE_FORMAT, command='bob pad evaluate'),
    criteria=CRITERIA)
def evaluate(ctx, scores, evaluation, **kwargs):
  common_options.evaluate_flow(
      ctx, scores, evaluation, metrics, roc, det, epc, hist, **kwargs)


@common_options.multi_metrics_command(
    common_options.MULTI_METRICS_HELP.format(
        names='FtA, APCER, BPCER, FAR, FRR, ACER',
        criteria=CRITERIA, score_format=SCORE_FORMAT,
        command='bob pad multi-metrics'),
    criteria=CRITERIA)
def multi_metrics(ctx, scores, evaluation, protocols_number, **kwargs):
  ctx.meta['min_arg'] = protocols_number * (2 if evaluation else 1)
  process = figure.MultiMetrics(
      ctx, scores, evaluation, load.split)
  process.run()
