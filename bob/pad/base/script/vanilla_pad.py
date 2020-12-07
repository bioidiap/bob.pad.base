"""Executes PAD pipeline"""


from bob.pipelines.distributed import VALID_DASK_CLIENT_STRINGS
import click
from bob.extension.scripts.click_helper import ConfigCommand
from bob.extension.scripts.click_helper import ResourceOption
from bob.extension.scripts.click_helper import verbosity_option
from bob.pipelines.distributed import dask_get_partition_size


@click.command(
    entry_point_group="bob.pad.config",
    cls=ConfigCommand,
    epilog="""\b
 Command line examples\n
 -----------------------


 $ bob pad vanilla-pad my_experiment.py -vv
""",
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
    entry_point_group="dask.client",
    string_exceptions=VALID_DASK_CLIENT_STRINGS,
    default="single-threaded",
    help="Dask client for the execution of the pipeline.",
    cls=ResourceOption,
)
@click.option(
    "--group",
    "-g",
    "groups",
    type=click.Choice(["dev", "eval"]),
    multiple=True,
    default=("dev", "eval"),
    help="If given, this value will limit the experiments belonging to a particular group",
)
@click.option(
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Saves scores (and checkpoints) in this folder.",
)
@click.option(
    "--checkpoint",
    "-c",
    is_flag=True,
    help="If set, it will checkpoint all steps of the pipeline. Checkpoints will be saved in `--output`.",
    cls=ResourceOption,
)
@click.option(
    "--dask-partition-size",
    "-s",
    help="If using Dask, this option defines the size of each dask.bag.partition."
    "Use this option if the current heuristic that sets this value doesn't suit your experiment."
    "(https://docs.dask.org/en/latest/bag-api.html?highlight=partition_size#dask.bag.from_sequence).",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@click.option(
    "--dask-n-workers",
    "-n",
    help="If using Dask, this option defines the number of workers to start your experiment."
    "Dask automatically scales up/down the number of workers due to the current load of tasks to be solved."
    "Use this option if the current amount of workers set to start an experiment doesn't suit you.",
    default=None,
    type=click.INT,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
@click.pass_context
def vanilla_pad(
    ctx,
    pipeline,
    database,
    dask_client,
    groups,
    output,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    **kwargs,
):
    """Runs the simplest PAD pipeline."""

    import gzip
    import logging
    import os
    import sys
    from glob import glob

    import bob.pipelines as mario
    import dask.bag
    from bob.extension.scripts.click_helper import log_parameters
    from bob.pipelines.distributed.sge import get_resource_requirements
    from bob.pipelines.utils import isinstance_nested
    from bob.pipelines.wrappers import DaskWrapper

    logger = logging.getLogger(__name__)
    log_parameters(logger)

    os.makedirs(output, exist_ok=True)

    if checkpoint:
        pipeline = mario.wrap(
            ["checkpoint"], pipeline, features_dir=output, model_path=output
        )

    # Fetching samples
    fit_samples = database.fit_samples()
    total_samples = len(fit_samples)
    predict_samples = dict()
    for group in groups:
        predict_samples[group] = database.predict_samples(group=group)
        total_samples += len(predict_samples[group])

    # Checking if the pipieline is dask-wrapped
    first_step = pipeline[0]
    if not isinstance_nested(first_step, "estimator", DaskWrapper):

        # Scaling up if necessary
        if dask_n_workers is not None and not isinstance(dask_client, str):
            dask_client.cluster.scale(dask_n_workers)

        # Defining the partition size
        partition_size = None
        if not isinstance(dask_client, str):
            lower_bound = 25  # lower bound of 25 videos per chunk
            partition_size = dask_get_partition_size(
                dask_client.cluster, total_samples, lower_bound=lower_bound
            )
        if dask_partition_size is not None:
            partition_size = dask_partition_size

        pipeline = mario.wrap(["dask"], pipeline, partition_size=partition_size)

    # create an experiment info file
    with open(os.path.join(output, "Experiment_info.txt"), "wt") as f:
        f.write(f"{sys.argv!r}\n")
        f.write(f"database={database!r}\n")
        f.write("Pipeline steps:\n")
        for i, name, estimator in pipeline._iter():
            f.write(f"Step {i}: {name}\n{estimator!r}\n")

    # train the pipeline
    pipeline.fit(fit_samples)

    for group in groups:

        logger.info(f"Running vanilla biometrics for group {group}")
        result = pipeline.decision_function(predict_samples[group])

        scores_path = os.path.join(output, f"scores-{group}")

        if isinstance(result, dask.bag.core.Bag):

            # write each partition into a zipped txt file
            result = result.map(pad_predicted_sample_to_score_line)
            prefix, postfix = f"{output}/scores/scores-{group}-", ".txt.gz"
            pattern = f"{prefix}*{postfix}"
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            logger.info("Writing bag results into files ...")
            resources = get_resource_requirements(pipeline)
            result.to_textfiles(
                pattern, last_endline=True, scheduler=dask_client, resources=resources
            )

            with open(scores_path, "w") as f:
                # concatenate scores into one score file
                for path in sorted(
                    glob(pattern),
                    key=lambda l: int(l.replace(prefix, "").replace(postfix, "")),
                ):
                    with gzip.open(path, "rt") as f2:
                        f.write(f2.read())
                    # delete intermediate score files
                    os.remove(path)

        else:
            with open(scores_path, "w") as f:
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
