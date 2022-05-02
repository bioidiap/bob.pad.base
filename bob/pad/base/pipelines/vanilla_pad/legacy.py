"""Re-usable blocks for legacy bob.pad.base databases"""

import logging

from bob.pipelines.sample import DelayedSample

from .abstract_classes import Database

logger = logging.getLogger(__name__)


def _padfile_to_delayed_sample(padfile, database):
    return DelayedSample(
        load=padfile.load,
        subject=str(padfile.client_id),
        attack_type=padfile.attack_type,
        key=padfile.path,
        delayed_attributes=dict(annotations=lambda: padfile.annotations),
        is_bonafide=padfile.attack_type is None,
    )


class DatabaseConnector(Database):
    """Wraps a bob.pad.base database and generates conforming samples

    This connector allows wrapping generic bob.pad.base datasets and generate samples
    that conform to the specifications of pad pipelines defined in this package.


    Parameters
    ----------
    database : object
        An instantiated version of a bob.pad.base.Database object
    """

    def __init__(
        self,
        database,
        annotation_type="eyes-center",
        fixed_positions=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.database = database
        self.annotation_type = annotation_type
        self.fixed_positions = fixed_positions

    def fit_samples(self):
        objects = self.database.training_files(flat=True)
        return [_padfile_to_delayed_sample(k, self.database) for k in objects]

    def predict_samples(self, group="dev"):
        objects = self.database.all_files(groups=group, flat=True)
        return [_padfile_to_delayed_sample(k, self.database) for k in objects]

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(
    database={self.database},
    annotation_type={self.annotation_type},
    fixed_positions={self.fixed_positions}
)"""
