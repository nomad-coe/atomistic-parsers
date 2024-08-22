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
from atomisticparsers.dlpoly import DLPolyParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return DLPolyParser()


def test_0(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/dl-poly-test1/OUTPUT', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '4.07'

    sec_method = sec_run.method[0]
    assert sec_method.x_dl_poly_control_parameters['real space cutoff'] == approx(
        1.2000e01
    )
    sec_atom_parameters = sec_method.molecule_parameters[0].atom_parameters
    assert len(sec_atom_parameters) == 2
    assert sec_atom_parameters[0].label == 'Na+'
    assert sec_atom_parameters[1].mass.magnitude == approx(5.88710915e-26)
    assert sec_atom_parameters[1].charge.magnitude == approx(-1.60217663e-19)
    assert sec_atom_parameters[0].x_dl_poly_nrept == 500
    sec_model = sec_method.force_field.model[0]
    assert len(sec_model.contributions) == 3
    assert (sec_model.contributions[0].atom_labels == ['Na+', 'Na+']).all()
    assert sec_model.contributions[1].functional_form == 'Born-Huggins-Meyer'
    assert sec_model.contributions[2].parameters[1] == approx(3.1545)

    sec_system = archive.run[0].system
    assert len(sec_system) == 3
    assert sec_system[0].atoms.lattice_vectors[2][2].magnitude == approx(9.878128e-09)
    assert sec_system[0].atoms.labels[10] == 'Na'
    assert np.shape(sec_system[0].atoms.positions) == (27000, 3)
    assert sec_system[0].atoms.positions[3][1].magnitude == approx(-1.98512856e-09)
    assert np.shape(sec_system[0].atoms.velocities) == (27000, 3)

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 22
    assert sec_calc[3].energy.total.value.magnitude == approx(-4.31824015e-10)
    assert sec_calc[14].energy.coulomb.value.magnitude == approx(-4.88158633e-10)
    assert sec_calc[8].energy.contributions[1].value.magnitude == approx(4.61213065e-11)
    assert sec_calc[17].temperature.magnitude == approx(503.57)
    assert sec_calc[9].pressure.magnitude == approx(-696315.532)
    assert sec_calc[6].x_dl_poly_virial_configurational == approx(1.5826e08)
    assert sec_calc[1].x_dl_poly_volume.to('angstrom ** 3').magnitude == approx(
        9.6388e05
    )
    assert sec_calc[2].time_physical.magnitude == approx(0.812)
    assert sec_calc[7].time_calculation.magnitude == approx(0.609)

    assert archive.workflow2.method.thermodynamic_ensemble == 'NVT'
    assert archive.workflow2.method.integration_timestep.magnitude == approx(1e-15)


def test_1(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench1/OUTPUT', archive, None)

    sec_system = archive.run[0].system
    assert len(sec_system) == 1
    assert len(sec_system[0].atoms.positions) == 19652
    assert sec_system[0].atoms.lattice_vectors[2][2].magnitude == approx(9.1885e-09)
    assert sec_system[0].atoms.labels[5] == 'Al'

    sec_model = archive.run[0].method[0].force_field.model[0]
    assert len(sec_model.contributions) == 1
    assert (sec_model.contributions[0].atom_labels == ['Al', 'Al']).all()
    assert sec_model.contributions[0].functional_form == 'Sutton-Chen'
    assert sec_model.contributions[0].parameters[4] == approx(16.399)


def test_2(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench2/OUTPUT', archive, None)

    sec_molecule_parameters = archive.run[0].method[0].molecule_parameters
    assert len(sec_molecule_parameters) == 2
    assert sec_molecule_parameters[1].label == 'water tip3p'
    assert sec_molecule_parameters[0].atom_parameters[20].charge.magnitude == approx(
        5.60761822e-21
    )

    sec_system = archive.run[0].system
    assert len(sec_system[0].constraint) == 127
    assert (sec_system[0].constraint[4].atom_indices == [9, 8]).all()
    assert sec_system[0].constraint[13].parameters[0] == approx(0.96000)
    assert (sec_system[0].constraint[126].atom_indices == [0, 1, 2]).all()


def test_3(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench3/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 27539
    assert len(sec_system[0].atoms.labels) == 27539
    assert len(sec_system[0].atoms.velocities) == 27539


def test_4(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench4/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 27000
    assert len(sec_system[0].atoms.labels) == 27000


def test_5(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench5/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 8640
    assert len(sec_system[0].atoms.labels) == 8640
    assert len(sec_system[0].atoms.velocities) == 8640


def test_6(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench6/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 3838
    assert len(sec_system[0].atoms.labels) == 3838
    assert len(sec_system[0].atoms.velocities) == 3838


def test_7(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench7/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 12390
    assert len(sec_system[0].atoms.labels) == 12390
    assert len(sec_system[0].atoms.velocities) == 12390


def test_8(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench8/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 5416
    assert len(sec_system[0].atoms.labels) == 5416
    assert len(sec_system[0].atoms.velocities) == 5416


def test_9(parser):
    archive = EntryArchive()

    parser.parse('tests/data/dlpoly/bench9/OUTPUT', archive, None)
    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.positions) == 18866
    assert len(sec_system[0].atoms.labels) == 18866
    assert len(sec_system[0].atoms.velocities) == 18866
