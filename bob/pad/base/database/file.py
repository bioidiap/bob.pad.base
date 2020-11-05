import bob.bio.base.database


class PadFile(bob.bio.base.database.BioFile):
    """A simple base class that defines basic properties of File object for the use in PAD experiments"""

    def __init__(
        self,
        client_id,
        path,
        attack_type=None,
        file_id=None,
        original_directory=None,
        original_extension=None,
        annotation_directory=None,
        annotation_extension=None,
        annotation_type=None,
    ):
        """Initialize the File object with the minimum required data.

        Parameters:

        attack_type : a string type
          In cased of a spoofed data, this parameter should indicate what kind of spoofed attack it is.
          The default None value is interpreted that the PadFile is a genuine or real sample.

        For client_id, path and file_id, please refer to :py:class:`bob.bio.base.database.BioFile` constructor
        """
        super(PadFile, self).__init__(
            client_id,
            path,
            file_id,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            annotation_type=annotation_type,
        )

        if attack_type is not None:
            assert isinstance(attack_type, str)

        # just copy the information
        # The attack type of the sample, None if it is a genuine sample.
        self.attack_type = attack_type
