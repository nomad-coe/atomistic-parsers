#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest

from nomad.datamodel import EntryArchive
from atomisticparsers.gulp import GulpParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return GulpParser()


def test_basic(parser):
    archive = EntryArchive()

    parser.parse('tests/data/gulp/example1.got', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '4.1.0'

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.lattice_vectors[2][1].magnitude == approx(-2.782858e-10)
    assert sec_system.atoms.labels[3] == 'O'
    assert sec_system.atoms.positions[1][1].magnitude == approx(-2.8417015e-11)

    sec_sccs = sec_run.calculation[0]
    assert sec_sccs.energy.total.value.magnitude == approx(-5.04963266e-17)


def test_1(parser):
    archive = EntryArchive()

    parser.parse('tests/data/gulp/example18.got', archive, None)

    sec_sccs = archive.run[0].calculation
    assert len(sec_sccs) == 4
    assert sec_sccs[2].energy.total.value.magnitude == approx(-2.10631951e-16)
