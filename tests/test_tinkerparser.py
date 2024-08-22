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
from atomisticparsers.tinker import TinkerParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return TinkerParser()


def test_minimize_dynamic(parser):
    archive = EntryArchive()

    parser.parse('tests/data/tinker/argon/argon.log', archive, None)

    sec_run = archive.run
    assert len(sec_run) == 2
    assert sec_run[0].program.version == '8.0'
    assert (
        sec_run[0]
        .x_tinker_section_control_parameters[0]
        .x_tinker_inout_control_vdw_cutoff
        == '12.0'
    )
    assert (
        sec_run[0].x_tinker_section_control_parameters[0].x_tinker_inout_control_lights
    )
    assert sec_run[0].x_tinker_control_parameters['randomseed'] == '123456789'

    assert len(sec_run[0].calculation) == 70
    assert sec_run[0].calculation[60].energy.total.value.magnitude == approx(
        -1.01001845e-16
    )

    assert len(sec_run[0].system) == 2
    assert sec_run[0].system[1].atoms.lattice_vectors[1][1].magnitude == approx(
        2.60206e-09
    )
    assert sec_run[0].system[1].atoms.labels[149] == 'Ar'
    assert len(sec_run[0].system[1].atoms.positions) == 150
    assert sec_run[0].system[1].atoms.positions[78][2].magnitude == approx(
        -1.0426974e-09
    )
    assert sec_run[0].calculation[-1].system_ref == sec_run[0].system[1]

    assert len(sec_run[1].calculation) == 6
    assert sec_run[1].calculation[3].energy.total.value.magnitude == approx(
        -8.20736128e-17
    )
    assert sec_run[1].calculation[4].temperature.magnitude == approx(163.21)
    assert sec_run[1].calculation[5].step == 3000

    sec_workflow = archive.workflow2
    # assert sec_workflow[0].type == 'geometry_optimization'
    # assert sec_workflow[0].geometry_optimization.method == 'Limited Memory BFGS Quasi-Newton'
    # assert sec_workflow[0].geometry_optimization.x_tinker_convergence_tolerance_rms_gradient == 1.0
    # assert sec_workflow[0].geometry_optimization.x_tinker_final_rms_gradient.magnitude == 0.9715

    assert sec_workflow.m_def.name == 'MolecularDynamics'
    assert sec_workflow.method.thermodynamic_ensemble is None
    assert sec_workflow.method.integration_timestep.magnitude == approx(2e-15)
    assert sec_workflow.x_tinker_number_of_steps_requested == 3000
    assert sec_workflow.x_tinker_barostat_tau == approx(1.0e20)
    assert sec_workflow.x_tinker_thermostat_target_temperature.magnitude == approx(
        150.87
    )
    assert sec_workflow.x_tinker_barostat_target_pressure.magnitude == approx(4964925.0)


def test_vibrate(parser):
    archive = EntryArchive()

    parser.parse('tests/data/tinker/enkephalin/enkephalin.log', archive, None)
    sec_run = archive.run
    assert len(sec_run) == 3

    assert len(sec_run[1].calculation) == 18
    assert sec_run[1].calculation[4].energy.total.value.magnitude == approx(
        -1.26219061e-16
    )

    assert len(sec_run[1].system) == 2
    assert sec_run[1].system[0].atoms.positions[60][0].magnitude == approx(
        -7.929526e-11
    )
    assert sec_run[1].system[1].atoms.positions[34][1].magnitude == approx(
        -5.2604576e-11
    )

    assert len(sec_run[2].system) == 1
    assert sec_run[2].system[0].atoms.positions[60][0].magnitude == approx(
        -7.5031232e-11
    )
    assert sec_run[2].calculation[0].vibrational_frequencies[0].value[
        200
    ].magnitude == approx(297486.3)
    assert sec_run[2].calculation[0].vibrational_frequencies[0].x_tinker_eigenvalues[
        79
    ] == approx(114.435)


def test_arc(parser):
    archive = EntryArchive()

    parser.parse('tests/data/tinker/ice/ice.log', archive, None)

    sec_run = archive.run
    assert len(sec_run) == 1
    assert sec_run[0].method[0].force_field.model[0].name == 'iwater'

    assert len(sec_run[0].calculation) == 2
    assert sec_run[0].calculation[0].energy.total.value.magnitude == approx(
        -1.75498075e-13
    )
    assert sec_run[0].calculation[0].energy.total.potential.magnitude == approx(
        -2.24844458e-13
    )
    assert sec_run[0].calculation[0].step == approx(100)
    assert sec_run[0].calculation[1].energy.total.kinetic.magnitude == approx(
        4.77713513e-14
    )
    assert sec_run[0].calculation[1].pressure.magnitude == approx(4.99210036e08)
    assert sec_run[0].calculation[1].temperature.magnitude == approx(254.89)

    assert len(sec_run[0].system) == 2
    assert sec_run[0].system[0].atoms.lattice_vectors[2][2].magnitude == approx(
        4.05187059e-09
    )
    assert len(sec_run[0].system[0].atoms.positions) == 3024
    assert sec_run[0].system[0].atoms.positions[42][2].magnitude == approx(
        -9.444748e-10
    )
    assert sec_run[0].system[1].atoms.labels[3017] == 'H'
    assert sec_run[0].system[1].atoms.positions[3002][1].magnitude == approx(
        -9.59405e-10
    )
    assert sec_run[0].calculation[1].system_ref == sec_run[0].system[1]
