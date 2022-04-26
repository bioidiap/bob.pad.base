#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.base.database.legacy import check_parameters_for_validity
from bob.pad.base.pipelines.vanilla_pad.abstract_classes import Database
from bob.pipelines.datasets import FileListDatabase, CSVToSamples


class CSVToPADSamples(CSVToSamples):
    """Converts a csv file to a list of PAD samples"""

    def __iter__(self):
        for sample in super().__iter__():
            if not hasattr(sample, "subject"):
                raise RuntimeError(
                    "PAD samples should contain a `subject` attribute which "
                    "reveals the identifies the person from whom the sample is created."
                )
            if not hasattr(sample, "attack_type"):
                raise RuntimeError(
                    "PAD samples should contain a `attack_type` attribute which "
                    "should be '' for bona fide samples and something like "
                    "print, replay, mask, etc. for attacks. This attribute is "
                    "considered the PAI type of each attack is used to compute APCER."
                )
            if sample.attack_type == "":
                sample.attack_type = None
            sample.is_bonafide = sample.attack_type is None
            if not hasattr(sample, "key"):
                sample.key = sample.filename
            yield sample


class FileListPadDatabase(Database, FileListDatabase):
    """A PAD database interface from CSV files."""

    def __init__(
        self,
        dataset_protocols_path,
        protocol,
        transformer=None,
        **kwargs,
    ):
        super().__init__(
            dataset_protocols_path=dataset_protocols_path,
            protocol=protocol,
            reader_cls=CSVToPADSamples,
            transformer=transformer,
            **kwargs,
        )

    def purposes(self):
        return ("real", "attack")

    def samples(self, groups=None, purposes=None):
        results = super().samples(groups=groups)
        purposes = check_parameters_for_validity(
            purposes, "purposes", self.purposes(), self.purposes()
        )

        def _filter(s):
            return (s.is_bonafide and "real" in purposes) or (
                (not s.is_bonafide) and "attack" in purposes
            )

        results = list(filter(_filter, results))
        return results

    def fit_samples(self):
        return self.samples(groups="train")

    def predict_samples(self, group="dev"):
        return self.samples(groups=group)
