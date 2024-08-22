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
from atomisticparsers.xtb import XTBParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return XTBParser()


def test_scf(parser):
    archive = EntryArchive()
    parser.parse('tests/data/xtb/scf_gfn2/out', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '6.4.1'
    assert sec_run.x_xtb_calculation_setup['coordinate file'] == 'coord'
    assert sec_run.x_xtb_calculation_setup['spin'] == approx(0.0)
    assert sec_run.time_run.date_start > 0
    assert sec_run.time_run.date_end > 0

    sec_method = sec_run.method
    assert sec_method[0].tb.name == 'xTB'
    assert sec_method[0].tb.x_xtb_setup['# basis functions'] == 6
    assert sec_method[0].tb.x_xtb_setup['Broyden damping'] == approx(0.4)

    sec_model = sec_method[0].tb.xtb
    assert sec_model.hamiltonian[0].parameters['H0-scaling (s, p, d)'][1] == approx(
        2.23
    )
    assert sec_model.contributions[0].type == 'dispersion'
    assert sec_model.contributions[0].parameters['a1'][0] == approx(0.52)
    assert sec_model.repulsion[0].parameters['rExp'][0] == approx(1.0)
    assert sec_model.coulomb[0].parameters['third order'][0] == 'shell-resolved'
    assert sec_model.coulomb[0].parameters['cn-shift'][0] == approx(1.2)

    sec_system = sec_run.system
    assert len(sec_system) == 1
    assert sec_system[0].atoms.positions[0][2].magnitude == approx(-3.8936111e-11)
    assert sec_system[0].atoms.labels[1] == 'H'

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 1
    assert sec_calc[0].energy.total.value.magnitude == approx(-2.21058735e-17)
    assert sec_calc[0].energy.x_xtb_scc.value.magnitude == approx(-2.22537188e-17)
    assert sec_calc[0].energy.x_xtb_anisotropic_xc.value.magnitude == approx(
        -3.54715078e-21
    )
    assert sec_calc[0].energy.x_xtb_dispersion.value.magnitude == approx(
        -6.15816388e-22
    )
    assert sec_calc[0].energy.x_xtb_repulsion.value.magnitude == approx(1.47845302e-19)
    assert len(sec_calc[0].scf_iteration) == 8
    assert sec_calc[0].scf_iteration[3].energy.total.value.magnitude == approx(
        -2.22536983e-17
    )
    assert sec_calc[0].scf_iteration[7].energy.change.magnitude == approx(
        -2.49449334e-27
    )
    assert sec_calc[0].eigenvalues[0].energies[0][0][2].magnitude == approx(
        -2.23973514e-18
    )
    assert sec_calc[0].eigenvalues[0].occupations[0][0][1] == approx(2.0)
    assert sec_calc[0].multipoles[0].dipole.total[2] == approx(7.53725637e-30)
    assert sec_calc[0].multipoles[0].dipole.x_xtb_q_only[2] == approx(5.27353596e-30)
    assert sec_calc[0].multipoles[0].quadrupole.total[5] == approx(-4.94417978e-40)
    assert sec_calc[0].multipoles[0].quadrupole.x_xtb_q_plus_dip[2] == approx(
        -2.42722437e-40
    )


def test_opt(parser):
    archive = EntryArchive()
    parser.parse('tests/data/xtb/opt_gfn1/out', archive, None)

    sec_run = archive.run[0]
    sec_system = sec_run.system
    assert len(sec_system) == 6
    assert sec_system[1].atoms.positions[0][2].magnitude == approx(-3.89361146e-11)
    assert sec_system[5].atoms.positions[2][0].magnitude == approx(-7.70994407e-11)

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 6
    assert sec_calc[1].energy.total.value.magnitude == approx(-2.51500371e-17)
    assert sec_calc[4].energy.change.magnitude == approx(-5.19400803e-25)
    assert len(sec_calc[2].scf_iteration) == 5
    assert sec_calc[3].scf_iteration[1].energy.total.value.magnitude == approx(
        -2.53150129e-17
    )
    assert sec_calc[0].time_calculation.magnitude == approx(0.002)
    assert sec_calc[1].time_physical.magnitude == approx(0.003)
    assert sec_calc[4].time_calculation.magnitude == approx(0.001)
    assert sec_calc[5].time_physical.magnitude == approx(0.007)

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == 'GeometryOptimization'
    assert (
        sec_workflow.method.convergence_tolerance_energy_difference.magnitude
        == approx(2.17987236e-23)
    )
    assert sec_workflow.method.convergence_tolerance_force_maximum.magnitude == approx(
        8.2387235e-11
    )
    assert sec_workflow.x_xtb_max_opt_cycles == 200
    assert sec_workflow.x_xtb_rf_solver == 'davidson'
    assert sec_workflow.x_xtb_hlow == approx(0.01)


def test_md(parser):
    archive = EntryArchive()
    parser.parse('tests/data/xtb/md/out', archive, None)

    sec_run = archive.run[0]
    sec_system = sec_run.system
    assert len(sec_system) == 201
    assert sec_system[4].atoms.positions[1][1].to('angstrom').magnitude == approx(
        0.01016905962746
    )
    assert sec_system[10].atoms.labels[7] == 'H'

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 51
    assert sec_calc[7].energy.total.value.magnitude == approx(-3.18464093e-17)
    assert sec_calc[13].energy.total.potential.magnitude == approx(-3.19064866e-17)
    assert sec_calc[20].energy.total.kinetic.magnitude == approx(8.23991752e-20)
    assert sec_calc[27].temperature.magnitude == approx(763.0)
    assert sec_calc[0].time_calculation.magnitude == approx(0.016)
    assert sec_calc[2].time_physical.magnitude == approx(0.16353535353535353)
    assert sec_calc[10].time_calculation.magnitude == approx(0.000734006734006734)
