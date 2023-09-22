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


def test_md(parser):
    archive = EntryArchive()

    parser.parse('tests/data/namd/apoa1-notraj/apoa1.log', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '2.12'
    assert sec_run.program.x_namd_build_osarch == 'Linux-x86_64-MPI'

    method = sec_run.method
    assert method[0].x_namd_input_parameters['PMEGridSizeX'] == 108
    assert method[0].x_namd_input_parameters['1-4scaling'] == 1.0
    assert method[0].x_namd_simulation_parameters['LDB PERIOD'] == 4000
    assert method[0].x_namd_simulation_parameters['PAIRLIST GROW RATE'] == 0.01

    system = sec_run.system
    assert len(system) == 2
    assert system[0].atoms.positions[80][1].magnitude == approx(9.629e-10)
    assert system[0].atoms.labels[57] == 'H'
    assert system[0].atoms.lattice_vectors[2][2].magnitude == approx(7.7758e-09)
    assert system[1].atoms.positions[8078][0].magnitude == approx(-1.780117e-09)

    calc = sec_run.calculation
    len(calc) == 501
    assert calc[37].energy.total.value.magnitude == approx(-1.42105253e-10)
    assert calc[0].energy.contributions[2].kind == 'dihedral'
    assert calc[302].energy.contributions[3].value.magnitude == approx(1.18452855e-13)
    assert calc[54].energy.total.kinetic.magnitude == approx(3.00344188e-11)
    assert calc[102].energy.total.potential.magnitude == approx(-1.72221646e-10)
    assert calc[487].energy.electronic.value.magnitude == approx(-2.15191687e-10)
    assert calc[468].energy.van_der_waals.value.magnitude == approx(1.49074358e-11)
    assert calc[416].pressure.magnitude == approx(-28276120.0)
    assert calc[21].temperature.magnitude == approx(152.9277)
    # TODO metainfo does not seem to apply unit for extended sections
    # assert calc[394].x_namd_pressure_average.magnitude == approx(-98841160.0)
    # assert calc[497].x_namd_volume.magnitude == approx(9.21491463e-25)
    assert calc[105].x_namd_temperature_average.magnitude == approx(147.9220)
    assert calc[148].time_calculation.magnitude == approx(0.215157)
    assert calc[76].time_physical.magnitude == approx(13.9035 + 0.217019 * 17)


def test_md_2(parser):
    archive = EntryArchive()

    parser.parse('tests/data/namd/er-gre-traj/er-gre.log', archive, None)

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.labels[48] == 'C'

    sec_scc = archive.run[0].calculation[101]
    assert sec_scc.energy.total.value.magnitude == approx(-2.38947258e-11)
