#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.pad.base.pipelines.vanilla_pad.abstract_classes import Database
import csv


from bob.pipelines.datasets.sample_loaders import CSVBaseSampleLoader
from bob.extension.download import search_file

from bob.pipelines import DelayedSample
import bob.io.base
import os
import functools


class CSVToSampleLoader(CSVBaseSampleLoader):
    """
    Simple mechanism that converts the lines of a CSV file to
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`

    Each CSV line of a PAD datasets should have at least a PATH and a SUBJECT id like
    in the example below:

    ```      
      .. code-block:: text

       PATH,SUBJECT
       path_1,subject_1
    ```

    """

    def check_header(self, header):
        """
        A header should have at least "subject" AND "PATH"
        """
        header = [h.lower() for h in header]
        if not "subject" in header:
            raise ValueError("The field `subject` is not available in your dataset.")

        if not "path" in header:
            raise ValueError("The field `path` is not available in your dataset.")

    def __call__(self, f, is_bonafide=True):
        f.seek(0)
        reader = csv.reader(f)
        header = next(reader)

        self.check_header(header)
        return [
            self.convert_row_to_sample(row, header, is_bonafide=is_bonafide)
            for row in reader
        ]

    def convert_row_to_sample(self, row, header=None, is_bonafide=True):
        path = str(row[0])
        subject = str(row[1])

        kwargs = dict([[str(h).lower(), r] for h, r in zip(header[2:], row[2:])])

        if self.metadata_loader is not None:
            metadata = self.metadata_loader(row, header=header, is_bonafide=is_bonafide)
            kwargs.update(metadata)

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            subject=subject,
            is_bonafide=is_bonafide,
            **kwargs,
        )


class LSTToSampleLoader(CSVBaseSampleLoader):
    """
    Simple mechanism that converts the lines of a LST file to
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`
    """

    def __call__(self, f, is_bonafide=True):
        f.seek(0)
        reader = csv.reader(f, delimiter=" ")
        samples = []

        for row in reader:
            if row[0][0] == "#":
                continue
            samples.append(self.convert_row_to_sample(row, is_bonafide=is_bonafide))

        return samples

    def convert_row_to_sample(self, row, header=None, is_bonafide=True):

        path = str(row[0])
        subject = str(row[1])
        attack_type = None
        if len(row) == 3:
            attack_type = str(row[2])

        kwargs = dict()
        if self.metadata_loader is not None:
            metadata = self.metadata_loader(row, header=header)
            kwargs.update(metadata)

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            subject=subject,
            is_bonafide=is_bonafide,
            attack_type=attack_type,
            **kwargs,
        )


class CSVPADDataset(Database):
    """
    Generic filelist dataset for PAD experiments.

    To create a new dataset, you need to provide a directory structure similar to the one below:

    .. code-block:: text

       my_dataset/
       my_dataset/my_protocol/train/for_real.csv
       my_dataset/my_protocol/train/for_attack.csv
       my_dataset/my_protocol/dev/for_real.csv
       my_dataset/my_protocol/dev/for_attack.csv
       my_dataset/my_protocol/eval/for_real.csv
       my_dataset/my_protocol/eval/for_attack.csv


    These csv files should contain in each row i-) the path to raw data and 
    ii-) and an identifier to the subject in the image (subject).    
    The structure of each CSV file should be as below:

    .. code-block:: text

       PATH,SUBJECT
       path_1,subject_1
       path_2,subject_2
       path_i,subject_j
       ...


    You might want to ship metadata within your Samples (e.g gender, age, annotations, ...)
    To do so is simple, just do as below:

    .. code-block:: text

       PATH,SUBJECT,TYPE_OF_ATTACK,GENDER,AGE
       path_1,subject_1,A,B,C
       path_2,subject_2,A,B,1
       path_i,subject_j,2,3,4
       ...


    The files `my_dataset/my_protocol/eval/for_real.csv` and `my_dataset/my_protocol/eval/for_attack.csv`
    are optional and it is used in case a protocol contains data for evaluation.

    Finally, the content of the files `my_dataset/my_protocol/train/for_real.csv` and `my_dataset/my_protocol/train/for_attack.csv` are used in the case a protocol
    contains data for training.

    Parameters
    ----------

        dataset_path: str
          Absolute path or a tarball of the dataset protocol description.

        protocol_na,e: str
          The name of the protocol

        csv_to_sample_loader: :any:`bob.bio.base.database.CSVBaseSampleLoader`
            Base class that whose objective is to generate :any:`bob.pipelines.Sample`
            and/or :any:`bob.pipelines.SampleSet` from csv rows
    

    """

    def __init__(
        self,
        dataset_protocol_path,
        protocol_name,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load,
            metadata_loader=None,
            dataset_original_directory="",
            extension="",
        ),
    ):
        self.dataset_protocol_path = dataset_protocol_path
        self.protocol_name = protocol_name

        def get_paths():

            if not os.path.exists(dataset_protocol_path):
                raise ValueError(f"The path `{dataset_protocol_path}` was not found")

            # Here we are handling the legacy
            train_real_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(protocol_name, "train", "for_real.lst"),
                    os.path.join(protocol_name, "train", "for_real.csv"),
                ],
            )

            train_attack_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(protocol_name, "train", "for_attack.lst"),
                    os.path.join(protocol_name, "train", "for_attack.csv"),
                ],
            )

            dev_real_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(protocol_name, "dev", "for_real.lst"),
                    os.path.join(protocol_name, "dev", "for_real.csv"),
                ],
            )

            dev_attack_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(protocol_name, "dev", "for_attack.lst"),
                    os.path.join(protocol_name, "dev", "for_attack.csv"),
                ],
            )

            eval_real_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(protocol_name, "eval", "for_real.lst"),
                    os.path.join(protocol_name, "eval", "for_real.csv"),
                ],
            )

            eval_attack_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(protocol_name, "eval", "for_attack.lst"),
                    os.path.join(protocol_name, "eval", "for_attack.csv"),
                ],
            )

            # The minimum required is to have `dev_enroll_csv` and `dev_probe_csv`

            # Dev
            if dev_real_csv is None:
                raise ValueError(
                    f"The file `{dev_real_csv}` is required and it was not found"
                )

            if dev_attack_csv is None:
                raise ValueError(
                    f"The file `{dev_attack_csv}` is required and it was not found"
                )

            return (
                train_real_csv,
                train_attack_csv,
                dev_real_csv,
                dev_attack_csv,
                eval_real_csv,
                eval_attack_csv,
            )

        (
            self.train_real_csv,
            self.train_attack_csv,
            self.dev_real_csv,
            self.dev_attack_csv,
            self.eval_real_csv,
            self.eval_attack_csv,
        ) = get_paths()

        def get_dict_cache():
            cache = dict()
            cache["train_real_csv"] = None
            cache["train_attack_csv"] = None
            cache["dev_real_csv"] = None
            cache["dev_attack_csv"] = None
            cache["eval_real_csv"] = None
            cache["eval_attack_csv"] = None
            return cache

        self.cache = get_dict_cache()
        self.csv_to_sample_loader = csv_to_sample_loader

    def _load_samples(self, cache_key, filepointer, is_bonafide):
        self.cache[cache_key] = (
            self.csv_to_sample_loader(filepointer, is_bonafide)
            if self.cache[cache_key] is None
            else self.cache[cache_key]
        )

        return self.cache[cache_key]

    def fit_samples(self):
        return self._load_samples(
            "train_real_csv", self.train_real_csv, is_bonafide=True
        ) + self._load_samples(
            "train_attack_csv", self.train_attack_csv, is_bonafide=False
        )

    def predict_samples(self, group="dev"):
        if group == "dev":
            return self._load_samples(
                "dev_real_csv", self.dev_real_csv, is_bonafide=True
            ) + self._load_samples(
                "dev_attack_csv", self.dev_attack_csv, is_bonafide=False
            )
        else:
            return self._load_samples(
                "eval_real_csv", self.eval_real_csv, is_bonafide=True
            ) + self._load_samples(
                "eval_attack_csv", self.eval_attack_csv, is_bonafide=False
            )
