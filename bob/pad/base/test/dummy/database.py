#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Thu Apr 21 16:41:21 CEST 2016
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import six

from bob.pad.db import PadFile
from bob.pad.db import PadDatabase

import bob.io.base
from bob.db.base.driver import Interface as BaseInterface

import pkg_resources
data_dir = pkg_resources.resource_filename('bob.pad.base', 'test/data')

dummy_name = "spoof_test"
dummy_train_list = ['train_real', 'train_attack']
dummy_devel_list = ['dev_real', 'dev_attack']
dummy_test_list = ['eval_real', 'eval_attack']

class TestFile(PadFile):
    def __init__(self, path, id):
        attack_type = None
        if "attack" in path:
            attack_type = "attack"
        PadFile.__init__(self, client_id=1, path=path, file_id=id, attack_type=attack_type)


def dumplist(args):
    """Dumps lists of files based on your criteria"""

    db = TestDatabase()
    data = db.get_all_data()
    output = sys.stdout

    if args.selftest:
        from bob.db.base.utils import null
        output = null()

    files = data[0] + data[1]
    for f in files:
        output.write('%s\n' % (f.make_path(args.directory, args.extension),))

    return 0


class Interface(BaseInterface):
    def name(self):
        return dummy_name

    def version(self):
        return '0.0.1'

    def files(self):
        files = dummy_train_list + dummy_devel_list + dummy_test_list
        return files

    def type(self):
        return 'rawfiles'

    def add_commands(self, parser):
        from argparse import SUPPRESS

        subparsers = self.setup_parser(parser,
                                       "Dummy Spoof Database", "Dummy spoof database with attacks for testing")

        dumpparser = subparsers.add_parser('dumplist', help="")
        dumpparser.add_argument('-d', '--directory', dest="directory", default='',
                            help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
        dumpparser.add_argument('-e', '--extension', dest="extension", default='',
                            help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
        dumpparser.add_argument('--self-test', dest="selftest", default=False,
                            action='store_true', help=SUPPRESS)

        dumpparser.set_defaults(func=dumplist)  # action


class TestDatabase(PadDatabase):
    """ Implements API of PAD DB interface for this Test database together with some low level support methods"""

    def __init__(self, protocol='Default', original_directory=data_dir, original_extension='', **kwargs):
        # call base class constructors to open a session to the database
        PadDatabase.__init__(self, name='testspoof', protocol=protocol,
                             original_directory=original_directory,
                             original_extension=original_extension, **kwargs)

    ################################################
    # Low level support methods for the database #
    ################################################
    def create_subparser(self, subparser, entry_point_name):
        from argparse import RawDescriptionHelpFormatter

        p = subparser.add_parser(entry_point_name,
                                 help=self.short_description(),
                                 description="Dummy description",
                                 formatter_class=RawDescriptionHelpFormatter)

        p.add_argument('--dummy-test', type=str, default='test',
                       dest="kwargs_protocol",
                       help='Test the functions of subparser')

    def get_protocols(self):
        return ["test"]

    def get_attack_types(self):
        return ["attack1", "attack2"]

    def name(self):
        i = Interface()
        return "Dummy Spoof Database (%s)" % i.name()

    def short_name(self):
        i = Interface()
        return i.name()

    def version(self):
        i = Interface()
        return i.version()

    def short_description(self):
        return "Dummy spoof database with attacks for testing"

    def long_description(self):
        return "Long description"

    def implements_any_of(self, propname):
        """
        Only support for audio files is implemented/
        :param propname: The type of data-support, which is checked if it contains 'spoof'
        :return: True if propname is None, it is equal to or contains 'spoof', otherwise False.
        """
        if isinstance(propname, (tuple, list)):
            return 'spoof' in propname
        elif propname is None:
            return True
        elif isinstance(propname, six.string_types):
            return 'spoof' == propname

        # does not implement the given access protocol
        return False
    def get_all_data(self):
        return self.all_files()

    #  This is the method from PadDatabase that we must implement
    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        fileset = []
        if purposes is None or 'real' in purposes:
            if groups is None or 'train' in groups:
                fileset += [TestFile(dummy_train_list[0], 1)]
            if groups is None or 'dev' in groups:
                fileset += [TestFile(dummy_devel_list[0], 2)]
            if groups is None or 'eval' in groups:
                fileset += [TestFile(dummy_test_list[0], 3)]
        if purposes is None or 'attack' in purposes:
            if groups is None or 'train' in groups:
                fileset += [TestFile(dummy_train_list[1], 4)]
            if groups is None or 'dev' in groups:
                fileset += [TestFile(dummy_devel_list[1], 5)]
            if groups is None or 'eval' in groups:
                fileset += [TestFile(dummy_test_list[1], 6)]
        return fileset


database = TestDatabase(original_directory=data_dir, original_extension='')
