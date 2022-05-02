"""Finalizes the scores that are produced by spoof.py
"""
import click

from bob.extension.scripts.click_helper import log_parameters, verbosity_option


@click.command(
    name="finalize-scores",
    epilog="""\b
Examples:
  $ bin/bob pad finalize_scores /path/to/scores-dev
  $ bin/bob pad finalize_scores /path/to/scores-{dev,eval}
""",
)
@click.argument(
    "scores", type=click.Path(exists=True, dir_okay=False), nargs=-1
)
@click.option(
    "-m",
    "--method",
    default="mean",
    type=click.Choice(["mean", "min", "max"]),
    show_default=True,
    help="The method to use when finalizing the scores.",
)
@click.option(
    "--backup/--no-backup", default=True, help="Whether to backup scores."
)
@verbosity_option()
def finalize_scores(scores, method, backup, verbose):
    """Finalizes the scores given by bob pad vanilla-pad
    When using bob.pad.base, Algorithms can produce several score values for
    each unique sample. You can use this script to average (or min/max) these
    scores to have one final score per sample.

    The conversion is done in-place (original files will be backed up).
    The order of scores will change.
    """
    import logging

    import numpy

    logger = logging.getLogger(__name__)
    log_parameters(logger)

    mean = {"mean": numpy.nanmean, "max": numpy.nanmax, "min": numpy.nanmin}[
        method
    ]

    for path in scores:
        new_lines = []
        with open(path) as f:
            old_lines = f.readlines()

        if backup:
            with open(f"{path}.bak", "w") as f:
                f.writelines(old_lines)

        old_lines.sort()

        for i, line in enumerate(old_lines):
            uniq, s = line.strip().rsplit(maxsplit=1)
            s = float(s)
            if i == 0:
                last_line = uniq
                last_scores = []

            if uniq == last_line:
                last_scores.append(s)
            else:
                new_lines.append("{} {}\n".format(last_line, mean(last_scores)))
                last_scores = [s]

            last_line = uniq

        else:  # this else is for the for loop
            new_lines.append("{} {}\n".format(last_line, mean(last_scores)))

        with open(path, "w") as f:
            f.writelines(new_lines)
