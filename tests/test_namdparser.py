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
from atomisticparsers.namd import NAMDParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return NAMDParser()


def test_basic(parser):
    archive = EntryArchive()

    parser.parse('tests/data/namd/apoa1-notraj/apoa1.log', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == 'NAMD 2.12'

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.positions[80][1].magnitude == approx(9.629e-10)
    assert sec_system.atoms.labels[57] == 'H'

    sec_sccs = sec_run.calculation
    len(sec_sccs) == 501
    assert sec_sccs[37].energy.total.value.magnitude == approx(-1.540870627208654e-15)


def test_1(parser):
    archive = EntryArchive()

    parser.parse('tests/data/namd/er-gre-traj/er-gre.log', archive, None)

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.labels[48] == 'C'

    sec_scc = archive.run[0].calculation[101]
    assert sec_scc.energy.total.value.magnitude == approx(-6.533433369963276e-16)
