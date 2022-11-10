"""Executes PAD pipeline"""


import logging

import click

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)
from bob.pipelines.distributed import (
    VALID_DASK_CLIENT_STRINGS,
    dask_get_partition_size,
)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.pad.config",
    cls=ConfigCommand,
    epilog="""\b
 Command line examples\n
 -----------------------


 $ bob pad run-pipeline my_experiment.py -vv
""",
)
@click.option(
    "--pipeline",
    "-p",
    required=True,
    entry_point_group="bob.pad.pipeline",
    help="Feature extraction algorithm",
    cls=ResourceOption,
)
@click.option(
    "--decision_function",
    "-f",
    show_default=True,
    default="decision_function",
    help="Name of the Pipeline step to call for results, eg. ``predict_proba``",
    cls=ResourceOption,
)
@click.option(
    "--database",
    "-d",
    required=True,
    entry_point_group="bob.pad.database",
    help="PAD Database connector (class that implements the methods: `fit_samples`, `predict_samples`)",
    cls=ResourceOption,
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
    type=click.Choice(["train", "dev", "eval"]),
    multiple=True,
    default=("dev", "eval"),
    help="If given, this value will limit the experiments belonging to a particular group",
    cls=ResourceOption,
)
@click.option(
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Saves scores (and checkpoints) in this folder.",
    cls=ResourceOption,
)
@click.option(
    "--checkpoint/--memory",
    "checkpoint",
    default=True,
    help="If --checkpoint (which is the default), all steps of the pipeline will be saved. Checkpoints will be saved in `--output`.",
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
@click.option(
    "--no-dask",
    is_flag=True,
    help="If set, it will not use Dask for the execution of the pipeline.",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def run_pipeline(
    pipeline,
    decision_function,
    database,
    dask_client,
    groups,
    output,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    no_dask,
    **kwargs,
):
    """Runs the simplest PAD pipeline."""

    from bob.extension.scripts.click_helper import log_parameters

    log_parameters(logger)

    execute_pipeline(
        pipeline=pipeline,
        database=database,
        decision_function=decision_function,
        output=output,
        groups=groups,
        checkpoint=checkpoint,
        dask_client=dask_client,
        dask_partition_size=dask_partition_size,
        dask_n_workers=dask_n_workers,
        no_dask=no_dask,
    )


def execute_pipeline(
    pipeline,
    database,
    decision_function="decision_function",
    output="results",
    groups=("dev", "eval"),
    checkpoint=False,
    dask_client="single-threaded",
    dask_partition_size=None,
    dask_n_workers=None,
    no_dask=False,
):
    import os
    import sys

    import dask.bag

    import bob.pipelines as mario

    from bob.pipelines import DaskWrapper, is_pipeline_wrapped
    from bob.pipelines.distributed.sge import get_resource_requirements

    if no_dask:
        dask_client = None

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

    # Checking if the pipeline is dask-wrapped
    if (
        not any(is_pipeline_wrapped(pipeline, DaskWrapper))
    ) and dask_client is not None:
        # Scaling up if necessary
        if dask_n_workers is not None and not isinstance(dask_client, str):
            dask_client.cluster.scale(dask_n_workers)

        # Defining the partition size
        partition_size = None
        if not isinstance(dask_client, str):
            lower_bound = 1  # lower bound of 1 video per chunk since usually video are already big
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
        logger.info(f"Running PAD pipeline for group {group}")
        result = getattr(pipeline, decision_function)(predict_samples[group])

        resources = None
        if isinstance(result, dask.bag.core.Bag):
            resources = get_resource_requirements(pipeline)

        save_sample_scores(
            result=result,
            output=output,
            group=group,
            dask_client=dask_client,
            resources=resources,
        )

    logger.info("PAD experiment finished!")


def _get_csv_columns(sample):
    """Returns a dict of {csv_column_name: sample_attr_name} given a sample."""
    # Mandatory columns and their corresponding fields
    columns_attr = {
        "claimed_id": "subject",
        "test_label": "key",
        "is_bonafide": "is_bonafide",
        "attack_type": "attack_type",
        "score": "data",
    }
    # Preventing duplicates and unwanted data
    ignored_fields = list(columns_attr.values()) + ["annotations"]
    # Retrieving custom metadata attribute names
    metadata_fields = [
        k
        for k in sample.__dict__.keys()
        if not k.startswith("_") and k not in ignored_fields
    ]
    for field in metadata_fields:
        columns_attr[field] = field
    return columns_attr


def sample_to_dict_row(sample, columns_fields):
    row_values = {
        col: getattr(sample, attr, None) for col, attr in columns_fields.items()
    }
    return row_values


def score_samples_to_dataframe(samples):
    import pandas as pd

    rows, column_fields = [], None
    for sample in samples:
        if column_fields is None:
            column_fields = _get_csv_columns(sample)
        row_values = sample_to_dict_row(sample, column_fields)
        rows.append(row_values)
    df = pd.DataFrame(rows)
    return df


def save_sample_scores(
    result,
    output,
    group,
    dask_client,
    resources=None,
):
    import os

    import dask.bag
    import dask.dataframe as dd

    scores_path = os.path.join(output, f"scores-{group}.csv")

    if isinstance(result, dask.bag.core.Bag):
        # convert score samples to dataframes
        result = result.map_partitions(score_samples_to_dataframe)
        result = dd.from_delayed(result.to_delayed())
        result.to_csv(
            scores_path,
            single_file=True,
            compute_kwargs=dict(scheduler=dask_client, resources=resources),
            index=False,
        )

    else:
        # convert score samples to dataframes
        result = score_samples_to_dataframe(result)
        result.to_csv(scores_path, index=False)
