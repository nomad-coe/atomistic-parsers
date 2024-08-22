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
import numpy as np

from nomad.datamodel import EntryArchive
from atomisticparsers.bopfox import BOPfoxParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return BOPfoxParser()


def test_energy(parser):
    archive = EntryArchive()
    parser.parse('tests/data/bopfox/Mo-W/log.bx', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '0.15 rev. 216'

    sec_method = sec_run.method[0]
    assert sec_method.x_bopfox_simulation_parameters['bopkernel'] == 'jackson'
    assert sec_method.x_bopfox_simulation_parameters['scftol'] == approx(0.001)
    sec_model = sec_method.tb.xtb
    assert sec_model.name == 'test'
    assert sec_model.hamiltonian[0].name == 'ddsigma'
    assert (sec_model.hamiltonian[1].atom_labels == ['W', 'W']).all()
    assert sec_model.hamiltonian[2].functional_form == 'screenedpowerlaw'
    assert sec_model.hamiltonian[3].x_bopfox_cutoff == approx(4.4)
    assert sec_model.hamiltonian[4].x_bopfox_dcutoff == approx(1.3)
    assert sec_model.hamiltonian[5].x_bopfox_valence == ['d', 'd']
    assert (sec_model.hamiltonian[6].atom_labels == ['W', 'Mo']).all()
    assert sec_model.hamiltonian[7].parameters[0] == approx(0.8359765)
    assert (sec_model.repulsion[2].atom_labels == ['Mo', 'Mo']).all()
    assert sec_model.repulsion[0].x_bopfox_cutoff == approx(5.0)
    assert sec_model.repulsion[1].parameters[2] == approx(-2.909290858)
    assert sec_model.repulsion[2].functional_form == 'env_Yukawa'
    assert sec_model.repulsion[3].x_bopfox_dcutoff == approx(0.0)

    sec_system = sec_run.system
    assert len(sec_system) == 1
    assert sec_system[0].atoms.labels[0] == 'W'
    assert sec_system[0].atoms.positions[1][2].magnitude == approx(1.55e-10)
    assert sec_system[0].atoms.lattice_vectors[0][0].magnitude == approx(3.1e-10)

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 1
    assert sec_calc[0].energy.total.value.magnitude == approx(-2.73845525e-18)
    assert sec_calc[0].energy.total.values_per_atom[1].magnitude == approx(
        -1.29324898e-18
    )
    assert sec_calc[0].energy.nuclear_repulsion.value.magnitude == approx(
        6.79146568e-21
    )
    assert sec_calc[0].energy.contributions[1].kind == 'prom'
    assert sec_calc[0].energy.contributions[1].value.magnitude == approx(0)
    assert sec_calc[0].energy.contributions[0].values_per_atom[0].magnitude == approx(
        -2.45798958e-18
    )
    assert sec_calc[0].energy.electrostatic.values_per_atom[0].magnitude == approx(0.0)
    assert sec_calc[0].energy.contributions[2].values_per_atom[1].magnitude == approx(
        1.72008688e-19
    )
    assert sec_calc[0].energy.contributions[3].value.magnitude == approx(1.78028693e-18)
    assert sec_calc[0].energy.contributions[4].values_per_atom[0].magnitude == approx(0)
    assert sec_calc[0].charges[0].value[0].magnitude == approx(-3.29865537e-20)
    assert sec_calc[0].charges[0].n_electrons[1] == approx(4.2058858813722590)
    assert sec_calc[0].x_bopfox_onsite_levels[0].orbital_projected[0].value == approx(
        -0.2068902284569173
    )
    assert sec_calc[0].x_bopfox_onsite_levels[0].orbital_projected[1].atom_index == 1
    assert sec_calc[0].x_bopfox_onsite_levels[0].orbital_projected[1].orbital == 'd'


def test_force(parser):
    archive = EntryArchive()
    parser.parse('tests/data/bopfox/FeC.fcc.distorted.CM/log.bx', archive, None)

    sec_calc = archive.run[0].calculation
    assert sec_calc[0].forces.total.value[1][0].magnitude == approx(-3.58311324e-11)
    assert sec_calc[0].forces.contributions[0].kind == 'analytic'
    assert sec_calc[0].forces.contributions[0].value[3][2].magnitude == approx(
        -2.56348261e-18
    )
    assert sec_calc[0].forces.contributions[2].value[2][0].magnitude == approx(
        -6.30558431e-11
    )
    assert sec_calc[0].stress.total.value[0][0].magnitude == approx(8.48866946e10)
    assert sec_calc[0].stress.total.value[2][0].magnitude == approx(-6879666.36)
    assert sec_calc[0].stress.total.values_per_atom[1][2][1].magnitude == approx(
        7.79970078e09
    )
    assert sec_calc[0].stress.contributions[0].values_per_atom[3][2][
        2
    ].magnitude == approx(1.8928643e12)
    assert sec_calc[0].stress.contributions[1].kind == 'rep1'
    assert sec_calc[0].stress.contributions[2].values_per_atom[2][0][
        0
    ].magnitude == approx(5.19185442e11)


def test_relaxation_verbose_notraj(parser):
    archive = EntryArchive()
    parser.parse('tests/data/bopfox/fcc.1x1x1.distorted/log.bx', archive, None)

    sec_system = archive.run[0].system
    assert len(sec_system) == 2
    assert sec_system[0].atoms.lattice_vectors[1][1].magnitude == approx(4.08e-10)
    assert sec_system[0].atoms.positions[2][0].magnitude == approx(2.04e-10)
    assert sec_system[1].atoms.positions[3][2].magnitude == approx(4.07999996e-10)

    sec_calc = archive.run[0].calculation
    assert len(sec_calc) == 23
    assert sec_calc[0].energy.total.values_per_atom[2].magnitude == approx(
        -1.31534881e-18
    )
    assert sec_calc[10].forces.total.value[1][0].magnitude == approx(-2.6174859e-12)


def test_relaxation_nonverbose_notraj(parser):
    archive = EntryArchive()
    parser.parse('tests/data/bopfox/v2_W.bcc.cell/log.bx', archive, None)

    sec_system = archive.run[0].system
    assert len(sec_system) == 2
    assert sec_system[0].atoms.lattice_vectors[2][2].magnitude == approx(3.163e-10)
    assert sec_system[0].atoms.positions[1][0].magnitude == approx(1.5815e-10)
    assert sec_system[1].atoms.positions[1][1].magnitude == approx(1.58272839e-10)

    sec_calc = archive.run[0].calculation
    assert len(sec_calc) == 48
    assert sec_calc[0].energy.contributions[1].value.magnitude == approx(2.89684962e-19)


def test_relaxation_verbose_traj(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/bopfox/fcc.2x2x2.vacancy.SCFreuseHii/log.bx', archive, None
    )

    sec_system = archive.run[0].system
    assert len(sec_system) == 5
    assert sec_system[0].atoms.positions[1][2].magnitude == approx(6.12e-10)
    assert sec_system[3].atoms.positions[8][1].magnitude == approx(4.08e-10)
    assert sec_system[4].atoms.positions[20][2].magnitude == approx(6.11684360e-10)

    sec_calc = archive.run[0].calculation
    assert len(sec_calc) == 5
    assert sec_calc[2].forces.contributions[0].value[2][0].magnitude == approx(
        -8.26723143e-17
    )


def test_md(parser):
    archive = EntryArchive()
    parser.parse('tests/data/bopfox/dimer/log.bx', archive, None)

    sec_system = archive.run[0].system
    assert len(sec_system) == 5
    assert sec_system[3].atoms.positions[1][0].magnitude == approx(2.187975e-10)

    sec_calc = archive.run[0].calculation
    assert len(sec_calc) == 200
    assert sec_calc[7].energy.total.value.magnitude == approx(-1.53363552e-18)
    assert sec_calc[12].energy.total.potential.magnitude == approx(-1.58580239e-18)
    assert sec_calc[18].energy.total.kinetic.magnitude == approx(9.72841652e-20)
    assert sec_calc[35].temperature.magnitude == approx(1895.1338)
    assert sec_calc[50].pressure.magnitude == approx(24236800.0)
