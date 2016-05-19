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

from bob.bio.base.preprocessor import Preprocessor
import os.path

dummy_data = {'train_real': 1.0, 'train_attack': 2.0,
              'dev_real': 3.0, 'dev_attack': 4.0,
              'eval_real': 5.0, 'eval_attack': 6.0}


class DummyPreprocessor(Preprocessor):
    def __init__(self, **kwargs):
        Preprocessor.__init__(self)

    def __call__(self, data, annotations):
        """Does nothing, simply converts the data type of the data, ignoring any annotation."""
        return data

    def read_original_data(self, original_file_name):
        return dummy_data[os.path.basename(original_file_name)]


preprocessor = DummyPreprocessor()
