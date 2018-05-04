"""Calculates PAD ISO compliant metrics based on the score files
"""
import click
from bob.measure.script import common_options
from bob.extension.scripts.click_helper import (verbosity_option,
                                                open_file_mode_option)
from bob.bio.base.score import load
from . import figure

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
    set and apply them on evaluation sets (if provided). The used thresholds are:

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

@click.command(context_settings=dict(token_normalize_func=lambda x: x.lower()))
@common_options.scores_argument(min_arg=2, force_eval=True, nargs=-1)
@common_options.eval_option()
@common_options.table_option()
@common_options.criterion_option()
@common_options.thresholds_option()
@open_file_mode_option()
@common_options.output_log_metric_option()
@common_options.legends_option()
@verbosity_option()
@click.pass_context
def vuln_metrics(ctx, scores, **kwargs):
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
