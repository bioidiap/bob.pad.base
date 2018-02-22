"""Calculates PAD ISO compliant metrics based on the score files
"""
import logging
import click
from bob.extension.scripts.click_helper import verbosity_option
from bob.measure.load import split
from bob.measure import (
    farfrr, far_threshold, eer_threshold, min_hter_threshold)

logger = logging.getLogger(__name__)
ALL_CRITERIA = ('bpcer20', 'eer', 'min-hter')


def scores_dev_eval(development_scores, evaluation_scores):
    dev_neg, dev_pos = split(development_scores)
    dev_neg.sort()
    dev_pos.sort()
    if evaluation_scores is None:
        logger.debug("No evaluation scores were provided.")
        eval_neg, eval_pos = None, None
    else:
        eval_neg, eval_pos = split(evaluation_scores)
        eval_neg.sort()
        eval_pos.sort()
    return dev_neg, dev_pos, eval_neg, eval_pos


def report(dev_neg, dev_pos, eval_neg, eval_pos, threshold):
    for group, neg, pos in [
        ('Development', dev_neg, dev_pos),
        ('Evaluation', eval_neg, eval_pos),
    ]:
        if neg is None:
            continue
        click.echo("{} set:".format(group))
        apcer, bpcer = farfrr(neg, pos, threshold)
        click.echo("APCER: {:>5.1f}%".format(apcer * 100))
        click.echo("BPCER: {:>5.1f}%".format(bpcer * 100))
        click.echo("HTER: {:>5.1f}%".format((apcer + bpcer) * 50))


@click.command(context_settings=dict(token_normalize_func=lambda x: x.lower()))
@click.argument('development_scores')
@click.argument('evaluation_scores', required=False)
@click.option(
    '-c', '--criterion', multiple=True, default=['bpcer20'],
    type=click.Choice(ALL_CRITERIA), help='The criteria to select. You can '
    'select multiple criteria by passing this option multiple times.',
    show_default=True)
@verbosity_option()
def metrics(development_scores, evaluation_scores, criterion):
    """PAD ISO compliant metrics.

    Reports several metrics based on a selected threshold on the development
    set. The thresholds are selected based on different criteria:

        bpcer20     When APCER is set to 5%.

        eer         When BPCER == APCER.

        min-hter    When HTER is minimum.

    Most metrics are according to the ISO/IEC 30107-3:2017 "Information
    technology -- Biometric presentation attack detection -- Part 3: Testing
    and reporting" standard. The reported metrics are:

        APCER: Attack Presentation Classification Error Rate

        BPCER: Bona-fide Presentation Classification Error Rate

        HTER (non-ISO): Half Total Error Rate ((BPCER+APCER)/2)

    Examples:

        $ bob pad metrics /path/to/scores-dev

        $ bob pad metrics /path/to/scores-dev /path/to/scores-eval

        $ bob pad metrics /path/to/scores-{dev,eval} # using bash expansion

        $ bob pad metrics -c bpcer20 -c eer /path/to/scores-dev
    """
    dev_neg, dev_pos, eval_neg, eval_pos = scores_dev_eval(
        development_scores, evaluation_scores)
    for method in criterion:
        if method == 'bpcer20':
            threshold = far_threshold(dev_neg, dev_pos, 0.05, True)
        elif method == 'eer':
            threshold = eer_threshold(dev_neg, dev_pos, True)
        elif method == 'min-hter':
            threshold = min_hter_threshold(dev_neg, dev_pos, True)
        else:
            raise ValueError("Unknown threshold criteria: {}".format(method))
        click.echo("\nThreshold of {} selected with the {} criteria".format(
            threshold, method))
        report(dev_neg, dev_pos, eval_neg, eval_pos, threshold)
