"""
Implementation of high-level interfaces for FileList-based databases that can be
used by both verification and PAD experiments.
"""

from bob.pad.base.database import PadFile
from bob.pad.base.database import FileListPadDatabase

from bob.bio.base.database import BioDatabase
from bob.bio.base.database.file import BioFile

import bob.io.base

import numpy
import scipy


class HighPadFile(PadFile):
    """
    A simple base class that defines basic properties of File object for the use in PAD experiments.
    Replace this class for the specific database.
    """

    def __init__(self, client_id, path, attack_type=None, file_id=None):
        """**Constructor Documentation**

        Initialize the Voice File object that can read WAV files.

        Parameters:

        For client_id, path, attack_type, and file_id, please refer
        to :py:class:`bob.pad.base.database.PadFile` constructor

        """

        super(HighPadFile, self).__init__(client_id, path, attack_type, file_id)

    def load(self, directory=None, extension='.wav'):
        path = self.make_path(directory, extension)
        # read audio
        if extension == '.wav':
            rate, audio = scipy.io.wavfile.read(path)
            # We consider there is only 1 channel in the audio file => data[0]
            return rate, numpy.cast['float'](audio)
        elif extension == '.avi':
            return bob.io.base.load(path)


class HighPadDatabase(FileListPadDatabase):
    def __init__(self,
                 original_directory="[DB_DATA_DIRECTORY]",
                 original_extension=".wav",
                 db_name='',
                 **kwargs):
        # call base class constructor
        from pkg_resources import resource_filename
        folder = resource_filename(__name__, '../lists/' + db_name)
        super(HighPadDatabase, self).__init__(folder, db_name, pad_file_class=HighPadFile,
                                              original_directory=original_directory,
                                              original_extension=original_extension,
                                              **kwargs)


class HighBioFile(BioFile):
    def __init__(self, f):
        """
        Initializes this File object with an File equivalent from the underlying SQl-based interface for
        database. Replace this class for the specific database.
        """
        super(HighBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)

        self.__f = f

    def load(self, directory=None, extension='.wav'):
        path = self.make_path(directory, extension)
        if extension == '.wav':
            rate, audio = scipy.io.wavfile.read(path)
            # We consider there is only 1 channel in the audio file => data[0]
            return rate, numpy.cast['float'](audio)
        elif extension == '.avi':
            return bob.io.base.load(path)


class HighBioDatabase(BioDatabase):
    """
    Implements verification API for querying High database.
    """

    def __init__(self,
                 original_directory="[DB_DATA_DIRECTORY]",
                 original_extension=".wav",
                 db_name='',
                 **kwargs):
        # call base class constructors to open a session to the database
        super(HighBioDatabase, self).__init__(name=db_name,
                                              original_directory=original_directory,
                                              original_extension=original_extension, **kwargs)

        self.__db = HighPadDatabase(db_name=db_name,
                                    original_directory=original_directory,
                                    original_extension=original_extension,
                                    **kwargs)

        self.low_level_group_names = ('train', 'dev', 'eval')
        self.high_level_group_names = ('world', 'dev', 'eval')

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)

        return [client.id for client in self.__db.clients(groups=groups, **kwargs)]

    def objects(self, protocol=None, purposes=None, model_ids=None, groups=None, **kwargs):
        """
        Maps objects method of PAD databases into objects method of
        Verification database

        Parameters
        ----------
        protocol : str
            To distinguish two vulnerability scenarios, protocol name should
            have either '-licit' or '-spoof' appended to it. For instance, if
            DB has protocol 'general', the named passed to this method should
            be 'general-licit', if we want to run verification experiments on
            bona fide data only, but it should be 'general-spoof', if we want
            to run it for spoof scenario (the probes are attacks).
        purposes : [str]
            This parameter is passed by the ``bob.bio.base`` verification
            experiment
        model_ids : [object]
            This parameter is passed by the ``bob.bio.base`` verification
            experiment
        groups : [str]
            We map the groups from ('world', 'dev', 'eval') used in
            verification experiments to ('train', 'dev', 'eval')
        **kwargs
            The rest of the parameters valid for a given database

        Returns
        -------
        [object]
            Set of BioFiles that verification experiments expect.

        """
        # convert group names from the conventional names in verification experiments to the internal database names
        if groups is None:  # all groups are assumed
            groups = self.high_level_group_names
        matched_groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)

        # this conversion of the protocol with appended '-licit' or '-spoof' is a hack for verification experiments.
        # To adapt spoofing databases to the verification experiments, we need to be able to split a given protocol
        # into two parts: when data for licit (only real/genuine data is used) and data for spoof
        # (attacks are used instead of real data) is used in the experiment.
        # Hence, we use this trick with appending '-licit' or '-spoof' to the
        # protocol name, so we can distinguish these two scenarios.
        # By default, if nothing is appended, we assume licit protocol.
        # The distinction between licit and spoof is expressed via purposes parameters, but
        # the difference is in the terminology only.

        # lets check if we have an appendix to the protocol name
        appendix = None
        if protocol:
            appendix = protocol.split('-')[-1]

        # if protocol was empty or there was no correct appendix, we just assume the 'licit' option
        if not (appendix == 'licit' or appendix == 'spoof'):
            appendix = 'licit'
        else:
            # put back everything except the appendix into the protocol
            protocol = '-'.join(protocol.split('-')[:-1])

        # if protocol was empty, we set it to the None
        if not protocol:
            protocol = None

        correct_purposes = purposes
        # licit protocol is for real access data only
        if appendix == 'licit':
            # by default we assume all real data, since this database has no enroll data
            if purposes is None:
                correct_purposes = ('real',)

        # spoof protocol uses real data for enrollment and spoofed data for probe
        # so, probe set is the same as attack set
        if appendix == 'spoof':
            # we return attack data only, since this database does not have explicit enroll data
            if purposes is None:
                correct_purposes = ('attack',)
            # otherwise replace 'probe' with 'attack'
            elif isinstance(purposes, (tuple, list)):
                correct_purposes = []
                for purpose in purposes:
                    if purpose == 'probe':
                        correct_purposes += ['attack']
                    else:
                        correct_purposes += [purpose]
            elif purposes == 'probe':
                correct_purposes = ('attack',)

        # now, query the underline PAD database
        objects = self.__db.objects(protocol=protocol, groups=matched_groups, purposes=correct_purposes, **kwargs)

        # make sure to return BioFile representation of a file, not the database one
        return [HighBioFile(f) for f in objects]

    def annotations(self, file):
        pass
