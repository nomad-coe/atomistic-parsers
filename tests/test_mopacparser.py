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
from atomisticparsers.mopac import MopacParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return MopacParser()


def test_basic(parser):
    archive = EntryArchive()

    parser.parse('tests/data/mopac/O2.out', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '15.347L'

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.positions[0][1].magnitude == approx(3e-10)
    assert sec_system.atoms.labels == ['O', 'O']

    sec_scc = sec_run.calculation[0]
    assert sec_scc.energy.total.value.magnitude == approx(-9.40492697e-17)
    assert sec_scc.forces.total.value[1][2].magnitude == approx(-1.0555523514531733e-08)


def test_1(parser):
    archive = EntryArchive()

    parser.parse('tests/data/mopac/C6H6.out', archive, None)

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.labels[6] == 'H'

    sec_scc = archive.run[0].calculation[0]
    assert sec_scc.energy.total.value.magnitude == approx(-1.30987804e-16)
