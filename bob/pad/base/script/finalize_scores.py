"""Finalizes the scores that are produced by bob pad run-pipeline.
"""
import click

from bob.extension.scripts.click_helper import log_parameters, verbosity_option


@click.command(
    name="finalize-scores",
    epilog="""\b
Examples:
  $ bob pad finalize_scores /path/to/scores-dev.csv
  $ bob pad finalize_scores /path/to/scores-{dev,eval}.csv
""",
)
@click.argument(
    "scores", type=click.Path(exists=True, dir_okay=False), nargs=-1
)
@click.option(
    "-m",
    "--method",
    default="mean",
    type=click.Choice(["mean", "median", "min", "max"]),
    show_default=True,
    help="The method to use when finalizing the scores.",
)
@click.option(
    "--backup/--no-backup", default=True, help="Whether to backup scores."
)
@verbosity_option()
def finalize_scores(scores, method, backup, verbose):
    """Finalizes the scores given by bob pad run-pipeline
    When using bob.pad.base, Algorithms can produce several score values for
    each unique sample. You can use this script to average (or min/max) these
    scores to have one final score per sample.

    The conversion is done in-place (original files will be backed up).
    The order of scores will change.
    """
    import logging
    import shutil

    import numpy
    import pandas as pd

    logger = logging.getLogger(__name__)
    log_parameters(logger)

    mean = {
        "mean": numpy.nanmean,
        "median": numpy.nanmedian,
        "max": numpy.nanmax,
        "min": numpy.nanmin,
    }[method]

    for path in scores:
        logger.info("Finalizing scores in %s", path)

        if backup:
            logger.info("Backing up %s", path)
            shutil.copy(path, path + ".bak")

        df = pd.read_csv(path)

        # average the scores of each frame
        df["score"] = df.groupby("video_key")["score"].transform(mean)

        # remove frame_id column if it exists
        if "frame_id" in df.columns:
            df.drop("frame_id", axis=1, inplace=True)

        # make rows unique based on test_label
        df.drop_duplicates(subset=["video_key"], inplace=True)

        df.to_csv(path, index=False)
