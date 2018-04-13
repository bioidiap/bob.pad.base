"""Generate random scores.
"""
import pkg_resources  # to make sure bob gets imported properly
import os
import logging
import numpy
import click
from click.types import FLOAT
from bob.extension.scripts.click_helper import verbosity_option
from bob.core import random
from bob.io.base import create_directories_safe

logger = logging.getLogger(__name__)

NUM_GENUINE_ACCESS = 5000
NUM_ZEIMPOSTORS = 5000
NUM_PA = 5000

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


def write_scores_to_file(pos, neg, filename, attack=False):
    """Writes score distributions into 4-column score files. For the format of
      the 4-column score files, please refer to Bob's documentation.

    Parameters
    ----------
    pos : array_like
        Scores for positive samples.
    neg : array_like
        Scores for negative samples.
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
