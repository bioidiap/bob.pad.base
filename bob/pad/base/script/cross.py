"""Prints Cross-db metrics analysis
"""
import os
import click
import logging
import yaml
import jinja2
from tabulate import tabulate
from bob.measure import eer_threshold, farfrr
from bob.measure.script import common_options
from bob.measure.utils import get_fta
from bob.extension.scripts.click_helper import verbosity_option
from bob.bio.base.score.load import split
from gridtk.generator import expand

logger = logging.getLogger(__name__)


@click.command(context_settings=dict(token_normalize_func=lambda x: x.lower()))
@click.argument('score_jinja_template')
@click.option('-d', '--database', 'databases', multiple=True, required=True,
              show_default=True,
              help='Names of the evaluation databases')
@click.option('-p', '--protocol', 'protocols', multiple=True, required=True,
              show_default=True,
              help='Names of the protocols of the evaluation databases')
@click.option('-a', '--algorithm', 'algorithms', multiple=True, required=True,
              show_default=True,
              help='Names of the algorithms')
@click.option('-td', '--train-database', required=True,
              help='The database that was used to train the algorithms.')
@click.option('-g', '--group', 'groups', multiple=True, show_default=True,
              default=['train', 'dev', 'eval'])
@common_options.table_option()
@common_options.output_log_metric_option()
@verbosity_option()
@click.pass_context
def cross(ctx, score_jinja_template, databases, protocols, algorithms,
          train_database, groups, **kwargs):
    """Cross-db analysis metrics
    """
    logger.debug('ctx.meta: %s', ctx.meta)
    logger.debug('score_jinja_template: %s', score_jinja_template)
    logger.debug('databases: %s', databases)
    logger.debug('protocols: %s', protocols)
    logger.debug('algorithms: %s', algorithms)
    logger.debug('train_database: %s', train_database)
    logger.debug('groups: %s', groups)
    logger.debug('kwargs: %s', kwargs)

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)

    data = {
        'evaluation': [{'database': db, 'protocol': proto}
                       for db, proto in zip(databases, protocols)],
        'algorithm': algorithms,
        'group': groups,
    }

    metrics = {}

    for variables in expand(yaml.dump(data, Dumper=yaml.SafeDumper)):
        logger.debug(variables)

        score_path = env.from_string(score_jinja_template).render(variables)
        logger.debug(score_path)

        database, protocol, algorithm, group = \
            variables['evaluation']['database'], \
            variables['evaluation']['protocol'], \
            variables['algorithm'], variables['group']

        # if algorithm name does not have train_database name in it.
        if train_database not in algorithm and database != train_database:
            score_path = score_path.replace(
                algorithm, database + '_' + algorithm)

        if not os.path.exists(score_path):
            metrics[(database, protocol, algorithm, group)] = \
                (float('nan'), ) * 5
            continue

        (neg, pos), fta = get_fta(split(score_path))

        if group == 'eval':
            threshold = metrics[(database, protocol, algorithm, 'dev')][1]
        else:
            threshold = eer_threshold(neg, pos)

        far, frr = farfrr(neg, pos, threshold)
        hter = (far + frr) / 2

        metrics[(database, protocol, algorithm, group)] = \
            (hter, threshold, fta, far, frr)

    logger.debug('metrics: %s', metrics)

    headers = ["Algorithms"] + list(databases)
    raws = []

    for algorithm in algorithms:
        raws.append([algorithm])
        for database, protocol in zip(databases, protocols):
            cell = ['{:>5.1f}'.format(
                100 * metrics[(database, protocol, algorithm, group)][0])
                for group in groups]
            raws[-1].append(' '.join(cell))

    title = ' Trained on {} '.format(train_database)
    title_line = '\n' + '=' * len(title) + '\n'
    click.echo(title_line + title + title_line, file=ctx.meta['log'])
    click.echo(tabulate(raws, headers, ctx.meta['tablefmt']),
               file=ctx.meta['log'])
