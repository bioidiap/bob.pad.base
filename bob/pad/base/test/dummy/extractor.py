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

from bob.bio.base.extractor import Extractor

_data = [0., 1., 2., 3., 4., 5., 6.]


class DummyExtractor(Extractor):
    def __init__(self, **kwargs):
        Extractor.__init__(self, requires_training=True)

    def __call__(self, data):
        """Does nothing, simply converts the data type of the data."""
        assert (data in _data)
        return data + 1.0

extractor = DummyExtractor()
