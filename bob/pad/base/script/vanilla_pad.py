"""Executes PAD pipeline"""


import csv
import logging

from io import StringIO

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
    "--decision_function",
    "-f",
    show_default=True,
    default="decision_function",
    help="Name of the Pipeline step to call for results, eg 'score' or 'predict'",
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
    "--csv-scores/--lst-scores",
    "write_metadata_scores",
    default=True,
    help="Choose the score file format as 'csv' with additional metadata or 'lst' 4 "
    "columns. Default: --csv-scores",
    cls=ResourceOption,
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
    decision_function,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    **kwargs,
):
    """Runs the simplest PAD pipeline."""

    from bob.extension.scripts.click_helper import log_parameters

    log_parameters(logger)

    compute_vanilla_pad(
        pipeline=pipeline,
        database=database,
        decision_function=decision_function,
        output=output,
        groups=groups,
        write_metadata_scores=write_metadata_scores,
        checkpoint=checkpoint,
        dask_client=dask_client,
        dask_partition_size=dask_partition_size,
        dask_n_workers=dask_n_workers,
    )


def compute_vanilla_pad(
    pipeline,
    database,
    decision_function="decision_function",
    output="results",
    groups=("dev", "eval"),
    write_metadata_scores=True,
    checkpoint=False,
    dask_client="single-threaded",
    dask_partition_size=None,
    dask_n_workers=None,
):
    import os
    import sys

    import bob.pipelines as mario

    from bob.pipelines.utils import isinstance_nested
    from bob.pipelines.wrappers import DaskWrapper

    get_score_row = (
        score_row_csv if write_metadata_scores else score_row_four_columns
    )
    output_file_ext = ".csv" if write_metadata_scores else ""
    intermediate_file_ext = ".csv.gz" if write_metadata_scores else ".txt.gz"

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
        result = getattr(pipeline, decision_function)(predict_samples[group])

        scores_path = os.path.join(output, f"scores-{group}{output_file_ext}")

        save_sample_scores(
            pipeline=pipeline,
            dask_client=dask_client,
            output=output,
            write_metadata_scores=write_metadata_scores,
            get_score_row=get_score_row,
            intermediate_file_ext=intermediate_file_ext,
            group=group,
            result=result,
            scores_path=scores_path,
        )


def save_sample_scores(
    pipeline,
    dask_client,
    output,
    write_metadata_scores,
    get_score_row,
    intermediate_file_ext,
    group,
    result,
    scores_path,
):

    import dask.bag

    from bob.pipelines.distributed.sge import get_resource_requirements

    if isinstance(result, dask.bag.core.Bag):
        resources = get_resource_requirements(pipeline)
        save_dask_sample_scores(
            result=result,
            get_score_row=get_score_row,
            output=output,
            group=group,
            intermediate_file_ext=intermediate_file_ext,
            write_metadata_scores=write_metadata_scores,
            scores_path=scores_path,
            dask_client=dask_client,
            resources=resources,
        )

    else:
        save_computed_sample_scores(
            result=result,
            get_score_row=get_score_row,
            write_metadata_scores=write_metadata_scores,
            scores_path=scores_path,
        )


def save_dask_sample_scores(
    result,
    get_score_row,
    output,
    group,
    intermediate_file_ext,
    write_metadata_scores,
    scores_path,
    dask_client,
    resources,
):
    import glob
    import gzip
    import os

    # write each partition into a zipped txt file, one line per sample
    result = result.map(get_score_row)
    prefix, postfix = f"{output}/scores/scores-{group}-", intermediate_file_ext
    pattern = f"{prefix}*{postfix}"
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    logger.info("Writing bag results into files ...")
    result.to_textfiles(
        pattern, last_endline=True, scheduler=dask_client, resources=resources
    )

    with open(scores_path, "w") as f:
        csv_writer, header = None, None
        # concatenate scores into one score file
        for path in sorted(
            glob(pattern),
            key=lambda l: int(l.replace(prefix, "").replace(postfix, "")),
        ):
            with gzip.open(path, "rt") as f2:
                if write_metadata_scores:
                    if csv_writer is None:
                        # Retrieve the header from one of the _header fields
                        tmp_reader = csv.reader(f2)
                        # Reconstruct a list from the str representation
                        header = next(tmp_reader)[-1].strip("][").split(", ")
                        header = [s.strip("' ") for s in header]
                        csv_writer = csv.DictWriter(f, fieldnames=header)
                        csv_writer.writeheader()
                        f2.seek(0, 0)
                        # There is no header in the intermediary files, specify it
                    csv_reader = csv.DictReader(
                        f2, fieldnames=header + ["_header"]
                    )
                    for row in csv_reader:
                        # Write each element of the row, except `_header`
                        csv_writer.writerow(
                            {k: row[k] for k in row.keys() if k != "_header"}
                        )
                else:
                    f.write(f2.read())
                    # delete intermediate score files
            os.remove(path)


def save_computed_sample_scores(
    result, get_score_row, write_metadata_scores, scores_path
):
    with open(scores_path, "w") as f:
        if write_metadata_scores:
            csv.DictWriter(
                f, fieldnames=_get_csv_columns(result[0]).keys()
            ).writeheader()
        for sample in result:
            f.write(get_score_row(sample, endl="\n"))


def score_row_four_columns(sample, endl=""):
    claimed_id, test_label, score = sample.subject, sample.key, sample.data

    # # use the model_label field to indicate frame number
    # model_label = getattr(sample, "frame_id", None)

    real_id = claimed_id if sample.is_bonafide else sample.attack_type

    if score is None:
        score = "nan"

    return f"{claimed_id} {real_id} {test_label} {score}{endl}"
    # return f"{claimed_id} {model_label} {real_id} {test_label} {score}{endl}"


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


def score_row_csv(sample, endl=""):
    """Returns a str representing one row of a CSV for the sample.

    If endl is empty, it is assumed that the row will be stored in a temporary file
    without header, thus a `_header` column is added at the end, containing the header
    as a list. This field can be used to reconstruct the final file.
    """
    columns_fields = _get_csv_columns(sample)
    string_stream = StringIO()
    csv_writer = csv.DictWriter(
        string_stream,
        fieldnames=list(columns_fields.keys())
        + (["_header"] if endl == "" else []),
    )

    row_values = {
        col: getattr(sample, attr, None) for col, attr in columns_fields.items()
    }
    if row_values["score"] is None:
        row_values["score"] = "nan"

    # Add a `_header` field to store the current CSV header (used in the dask Bag case)
    if endl == "":
        row_values["_header"] = list(columns_fields.keys())

    csv_writer.writerow(row_values)
    out_string = string_stream.getvalue()
    if endl == "":
        return out_string.rstrip()
    else:
        return out_string
