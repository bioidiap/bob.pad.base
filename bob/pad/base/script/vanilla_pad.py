"""Executes PAD pipeline"""

import logging
from os import pipe

import bob.pipelines as mario
import click
from bob.extension.scripts.click_helper import (ConfigCommand, ResourceOption,
                                                verbosity_option)

logger = logging.getLogger(__name__)


EPILOG = """\b


 Command line examples\n
 -----------------------


 $ bob pad vanilla-pad my_experiment.py -vv


 my_experiment.py must contain the following elements:

 >>> preprocessor = my_preprocessor() \n
 >>> extractor = my_extractor() \n
 >>> algorithm = my_algorithm() \n
 >>> checkpoints = EXPLAIN CHECKPOINTING \n

\b


Look at the following example

 $ bob pipelines vanilla-biometrics ./bob/pipelines/config/distributed/sge_iobig_16cores.py \
                                    ./bob/pipelines/config/database/mobio_male.py \
                                    ./bob/pipelines/config/baselines/facecrop_pca.py

\b



TODO: Work out this help

"""


@click.command(
    entry_point_group="bob.pad.config",
    cls=ConfigCommand,
    epilog=EPILOG,
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    entry_point_group="sklearn.pipeline",
    help="Feature extraction algorithm",
    cls=ResourceOption,
)
@click.option(
    "--database",
    "-d",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.pad.database",
    help="PAD Database connector (class that implements the methods: `fit_samples`, `predict_samples`)",
)
@click.option(
    "--dask-client",
    "-l",
    required=False,
    cls=ResourceOption,
    help="Dask client for the execution of the pipeline.",
)
@click.option(
    "--group",
    "-g",
    "groups",
    type=click.Choice(["dev", "eval"]),
    multiple=True,
    default=("dev",),
    help="If given, this value will limit the experiments belonging to a particular group",
)
@click.option(
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Name of output directory",
)
@click.option(
    "--checkpoint",
    "-c",
    is_flag=True,
    help="If set, it will checkpoint all steps of the pipeline. Checkpoints will be saved in `--output`.",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def vanilla_pad(pipeline, database, dask_client, groups, output, checkpoint, **kwargs):
    """Runs the simplest PAD pipeline.

    Such pipeline consists into three sub-pipelines.
    In all of them, given raw data as input it does the following steps:

    Sub-pipeline 1:\n
    ---------------

    Training background model. Some biometric algorithms demands the training of background model, for instance, PCA/LDA matrix or a Neural networks. This sub-pipeline handles that and it consists of 3 steps:

    \b
    raw_data --> preprocessing >> feature extraction >> train background model --> background_model



    \b

    Sub-pipeline 2:\n
    ---------------

    Creation of biometric references: This is a standard step in a biometric pipelines.
    Given a set of samples of one identity, create a biometric reference (a.k.a template) for sub identity. This sub-pipeline handles that in 3 steps and they are the following:

    \b
    raw_data --> preprocessing >> feature extraction >> enroll(background_model) --> biometric_reference

    Note that this sub-pipeline depends on the previous one



    Sub-pipeline 3:\n
    ---------------


    Probing: This is another standard step in biometric pipelines. Given one sample and one biometric reference, computes a score. Such score has different meanings depending on the scoring method your biometric algorithm uses. It's out of scope to explain in a help message to explain what scoring is for different biometric algorithms.


    raw_data --> preprocessing >> feature extraction >> probe(biometric_reference, background_model) --> score

    Note that this sub-pipeline depends on the two previous ones


    """

    import gzip
    import os
    from glob import glob

    import dask.bag

    os.makedirs(output, exist_ok=True)

    if checkpoint:
        pipeline = mario.wrap(
            ["checkpoint"], pipeline, features_dir=output, model_path=output
        )

    if dask_client is not None:
        pipeline = mario.wrap(["dask"], pipeline)

    # train the pipeline
    fit_samples = database.fit_samples()  # [::50]
    pipeline = pipeline.fit(fit_samples)

    for group in groups:

        logger.info(f"Running vanilla biometrics for group {group}")
        predict_samples = database.predict_samples(group=group)  # [::50]
        result = pipeline.decision_function(predict_samples)

        with open(os.path.join(output, f"scores-{group}"), "w") as f:

            if isinstance(result, dask.bag.core.Bag):
                if dask_client is None:
                    logger.warning(
                        "`dask_client` not set. Your pipeline will run locally"
                    )

                # write each partition into a zipped txt file
                result = result.map(pad_predicted_sample_to_score_line)
                prefix, postfix = f"{output}/scores/scores-{group}-", ".txt.gz"
                pattern = f"{prefix}*{postfix}"
                os.makedirs(os.path.dirname(prefix), exist_ok=True)
                logger.info("Writing bag results into files ...")
                result.to_textfiles(pattern, last_endline=True, scheduler=dask_client)

                # concatenate scores into one score file
                for path in sorted(
                    glob(pattern),
                    key=lambda l: int(l.replace(prefix, "").replace(postfix, "")),
                ):
                    with gzip.open(path, "rt") as f2:
                        f.write(f2.read())

            else:
                for sample in result:
                    f.write(pad_predicted_sample_to_score_line(sample, endl="\n"))


def pad_predicted_sample_to_score_line(sample, endl=""):
    claimed_id, test_label, score = sample.subject, sample.key, sample.data

    # # use the model_label field to indicate frame number
    # model_label = None
    # if hasattr(sample, "frame_id"):
    #     model_label = sample.frame_id

    real_id = claimed_id if sample.is_bonafide else sample.attack_type

    return f"{claimed_id} {real_id} {test_label} {score}{endl}"
    # return f"{claimed_id} {model_label} {real_id} {test_label} {score}{endl}"
