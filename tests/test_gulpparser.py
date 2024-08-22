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


def test_optimise_conp_property_old(parser):
    archive = EntryArchive()

    parser.parse('tests/data/gulp/example1_old/example1.got', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '4.1.0'
    assert sec_run.x_gulp_title == 'alumina test file'
    assert sec_run.x_gulp_n_cpu == 1
    assert sec_run.x_gulp_host_name == 'M-A0002884.local'
    assert sec_run.time_run.date_start > 0
    assert sec_run.time_run.date_end > 0

    sec_method = sec_run.method[0]
    contributions = sec_method.force_field.model[0].contributions
    assert len(contributions) == 4
    assert contributions[0].functional_form == 'Buckingham'
    assert contributions[1].atom_labels[0][1] == 'O'
    assert contributions[2].parameters['A'] == approx(404.0)
    assert contributions[3].parameters['cutoff_max'] == approx(0.8)
    assert len(sec_method.atom_parameters) == 4
    assert sec_method.atom_parameters[0].label == 'Al'
    assert sec_method.atom_parameters[1].x_gulp_type == 'shell'
    assert sec_method.atom_parameters[2].atom_number == 8
    assert sec_method.atom_parameters[3].mass.magnitude == approx(0.0)
    assert sec_method.atom_parameters[0].charge.magnitude == approx(6.88935953e-21)
    assert sec_method.atom_parameters[1].x_gulp_covalent_radius.magnitude == approx(
        1.35e-10
    )
    assert sec_method.atom_parameters[2].x_gulp_ionic_radius.magnitude == approx(0.0)
    assert sec_method.atom_parameters[3].x_gulp_vdw_radius.magnitude == approx(1.36e-10)

    sec_system = sec_run.system
    assert len(sec_system) == 2
    assert len(sec_system[0].atoms.positions) == 10
    assert sec_system[0].atoms.lattice_vectors[2][1].magnitude == approx(
        -2.74830275e-10
    )
    assert sec_system[0].atoms.labels[1] == 'O'
    assert sec_system[0].atoms.positions[1][1].magnitude == approx(-1.11691024e-11)
    assert sec_system[1].atoms.lattice_vectors[1][0].magnitude == approx(-2.410026e-10)
    assert sec_system[1].atoms.positions[0][2].magnitude == approx(7.59013967e-10)
    assert sec_system[1].atoms.periodic == [True, True, True]

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 2
    assert sec_calc[0].energy.total.value.magnitude == approx(-5.04201973e-17)
    assert sec_calc[0].energy.x_gulp_interatomic_potentials.magnitude == approx(
        1.01073584e-17
    )
    assert sec_calc[0].energy.x_gulp_monopole_monopole_real.magnitude == approx(
        -2.03424558e-17
    )
    assert sec_calc[0].energy.x_gulp_monopole_monopole_recip.magnitude == approx(
        -4.01850999e-17
    )
    assert sec_calc[1].energy.x_gulp_monopole_monopole_total.magnitude == approx(
        -6.00279419e-17
    )
    assert sec_calc[1].energy.total.value.magnitude == approx(-5.04963266e-17)
    assert sec_calc[1].x_gulp_piezoelectric_strain_matrix[1][0].magnitude == approx(0.0)
    assert sec_calc[1].x_gulp_piezoelectric_stress_matrix[1][0].magnitude == approx(0.0)
    assert sec_calc[1].x_gulp_static_dielectric_constant_tensor[2][2] == approx(
        16.36434
    )
    assert sec_calc[1].x_gulp_high_frequency_dielectric_constant_tensor[1][1] == approx(
        5.63011
    )
    assert sec_calc[1].x_gulp_static_refractive_indices[1] == approx(3.36052)
    assert sec_calc[1].x_gulp_high_frequency_refractive_indices[0] == approx(2.37279)
    sec_opt = sec_calc[0].x_gulp_bulk_optimisation
    assert sec_opt.x_gulp_n_variables == 6
    assert sec_opt.x_gulp_max_n_calculations == 1000
    assert sec_opt.x_gulp_max_hessian_update_interval == 10
    assert sec_opt.x_gulp_max_step_size == approx(1.0)
    assert sec_opt.x_gulp_max_parameter_tolerance == approx(0.00001)
    assert sec_opt.x_gulp_max_function_tolerance == approx(0.00001)
    assert sec_opt.x_gulp_max_gradient_tolerance == approx(0.001)
    assert sec_opt.x_gulp_max_gradient_component == approx(0.01)

    sec_workflow = archive.workflow2
    assert sec_workflow.results.elastic_constants_matrix_second_order[1][
        3
    ].magnitude == approx(-4.45913e10)
    assert sec_workflow.results.compliance_matrix_second_order[4][
        5
    ].magnitude == approx(-3.838e-12)
    assert sec_workflow.results.bulk_modulus_voigt.magnitude == approx(3.5517284e11)
    assert sec_workflow.results.bulk_modulus_reuss.magnitude == approx(3.5201356e11)
    assert sec_workflow.results.shear_modulus_hill.magnitude == approx(1.264132e11)
    assert sec_workflow.results.x_gulp_velocity_s_wave_reuss.magnitude == approx(
        5337.12
    )
    assert sec_workflow.results.x_gulp_velocity_p_wave_hill.magnitude == approx(
        11600.06
    )
    assert sec_workflow.results.x_gulp_compressibility.magnitude == approx(2.8408e-12)
    assert sec_workflow.results.x_gulp_youngs_modulus_y.magnitude == approx(
        3.7496954e11
    )
    assert sec_workflow.results.x_gulp_poissons_ratio[0][2] == approx(0.19595)


def test_single_md_conv_old(parser):
    archive = EntryArchive()

    parser.parse('tests/data/gulp/example18_old/example18.got', archive, None)

    sec_system = archive.run[0].system
    assert len(sec_system) == 1
    assert len(sec_system[0].atoms.positions) == 64
    assert sec_system[0].atoms.lattice_vectors[1][1].magnitude == approx(8.423972e-10)
    assert sec_system[0].atoms.positions[46][0].magnitude == approx(6.317979e-10)
    assert sec_system[0].atoms.periodic == [True, True, True]

    sec_calc = archive.run[0].calculation
    assert len(sec_calc) == 5
    assert sec_calc[2].energy.total.value.magnitude == approx(-1.91632663e-16)
    assert sec_calc[4].energy.x_gulp_total_averaged.value.magnitude == approx(
        -2.05507335e-16
    )
    assert sec_calc[1].energy.total.kinetic.magnitude == approx(3.82803257e-19)
    assert sec_calc[3].energy.total.potential.magnitude == approx(-2.1107173e-16)
    assert sec_calc[2].energy.x_gulp_total_averaged.kinetic.magnitude == approx(
        5.07815332e-19
    )
    assert sec_calc[1].time.magnitude == approx(5e-15)
    assert sec_calc[4].temperature.magnitude == approx(291.528999)
    assert sec_calc[1].pressure.magnitude == approx(5.56407e08)
    assert sec_calc[2].x_gulp_temperature_averaged.magnitude == approx(389.215575)
    assert sec_calc[3].x_gulp_pressure_averaged.magnitude == approx(1.2832591e10)

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == 'MolecularDynamics'
    assert sec_workflow.method.thermodynamic_ensemble == 'NVT'
    assert sec_workflow.method.integration_timestep.magnitude == approx(1e-15)
    assert sec_workflow.x_gulp_production_time.magnitude == approx(5e-13)
    assert sec_workflow.x_gulp_td_field_start_time.magnitude == approx(0.0)
    assert sec_workflow.x_gulp_n_degrees_of_freedom == approx(189)
    assert sec_workflow.x_gulp_friction_temperature_bath == approx(0.1)


def test_opti_mole_defe(parser):
    archive = EntryArchive()

    parser.parse('tests/data/gulp/example8/example8.got', archive, None)

    sec_model = archive.run[0].method[0].force_field.model
    assert len(sec_model) == 4
    assert sec_model[0].contributions[1].functional_form == 'Spring (c-s)'
    assert sec_model[1].contributions[0].atom_labels[0][1] == 'O'
    assert sec_model[2].contributions[0].parameters['Buckingham rho'] == approx(
        0.19760000
    )

    sec_system = archive.run[0].system
    assert len(sec_system) == 3
    assert len(sec_system[0].atoms.positions) == 24
    assert sec_system[0].atoms.positions[18][0].magnitude == approx(5.09156424e-10)
