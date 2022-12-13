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
from atomisticparsers.lammps import LammpsParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return LammpsParser()


def test_nvt(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/hexane_cyclohexane/log.hexane_cyclohexane_nvt', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '14 May 2016'

    sec_workflow = archive.workflow[0]
    section_md = sec_workflow.molecular_dynamics
    assert sec_workflow.type == 'molecular_dynamics'
    assert section_md.thermodynamic_ensemble == 'NVT'
    assert section_md.finished_normally is False
    assert section_md.with_trajectory is True
    assert section_md.with_thermodynamics is True
    assert section_md.integration_parameters.integrator_type == 'velocity_verlet'
    assert section_md.integration_parameters.integration_timestep.magnitude == 2.5e-16
    assert section_md.integration_parameters.integration_timestep.units == 'second'
    assert section_md.integration_parameters.n_steps == 80000
    assert section_md.integration_parameters.coordinate_save_frequency == 400
    assert section_md.integration_parameters.thermodynamics_save_frequency == 400
    assert section_md.integration_parameters.thermostat_parameters.thermostat_type == 'nose_hoover'
    assert section_md.integration_parameters.thermostat_parameters.reference_temperature.magnitude == 300.0
    assert section_md.integration_parameters.thermostat_parameters.reference_temperature.units == 'kelvin'
    assert section_md.integration_parameters.thermostat_parameters.coupling_constant.magnitude == 2.5e-14
    assert section_md.integration_parameters.thermostat_parameters.coupling_constant.units == 'second'

    sec_method = sec_run.method[0]
    assert len(sec_method.force_field.model[0].contributions) == 4
    assert sec_method.force_field.model[0].contributions[2].type == 'harmonic'
    assert sec_method.force_field.model[0].contributions[0].parameters[0][2] == 0.066
    assert sec_method.force_field.force_calculations.coulomb_cutoff.magnitude == 1.2000000000000002e-08
    assert sec_method.force_field.force_calculations.coulomb_cutoff.units == 'meter'
    assert sec_method.force_field.force_calculations.neighbor_searching.neighbor_update_frequency == 10

    sec_system = sec_run.system
    assert len(sec_system) == 201
    assert sec_system[5].atoms.lattice_vectors[1][1].magnitude == approx(2.24235e-09)
    assert False not in sec_system[0].atoms.periodic
    assert sec_system[80].atoms.labels[91:96] == ['H', 'H', 'H', 'C', 'C']

    sec_scc = sec_run.calculation
    assert len(sec_scc) == 201
    assert sec_scc[21].energy.current.value.magnitude == approx(8.86689197e-18)
    assert sec_scc[180].time_calculation.magnitude == 218.5357
    assert sec_scc[56].pressure.magnitude == approx(-77642135.4975)
    assert sec_scc[103].temperature.magnitude == 291.4591
    assert sec_scc[11].step == 4400
    assert len(sec_scc[1].energy.contributions) == 9
    assert sec_scc[112].energy.contributions[8].kind == 'kspace long range'
    assert sec_scc[96].energy.contributions[2].value.magnitude == approx(1.19666271e-18)
    assert sec_scc[47].energy.contributions[4].value.magnitude == approx(1.42166035e-18)

    assert sec_run.x_lammps_section_control_parameters[0].x_lammps_inout_control_atomstyle == 'full'


def test_thermo_format(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/1_methyl_naphthalene/log.1_methyl_naphthalene', archive, None)

    sec_sccs = archive.run[0].calculation
    assert len(sec_sccs) == 301
    assert sec_sccs[98].energy.total.value.magnitude == approx(1.45322428e-17)

    assert len(archive.run[0].system) == 4


def test_traj_xyz(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/methane_xyz/log.methane_nvt_traj_xyz_thermo_style_custom', archive, None)

    sec_systems = archive.run[0].system
    assert len(sec_systems) == 201
    assert sec_systems[13].atoms.positions[7][0].magnitude == approx(-8.00436e-10)


def test_traj_dcd(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/methane_dcd/log.methane_nvt_traj_dcd_thermo_style_custom', archive, None)

    assert len(archive.run[0].calculation) == 201
    sec_systems = archive.run[0].system
    assert np.shape(sec_systems[56].atoms.positions) == (320, 3)
    assert len(sec_systems[107].atoms.labels) == 320


def test_unwrapped_pos(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/1_xyz_files/log.lammps', archive, None)

    assert len(archive.run[0].calculation) == 101
    sec_systems = archive.run[0].system
    assert sec_systems[1].atoms.positions[452][2].magnitude == approx(5.99898)  # JFR - units are incorrect?!
    assert sec_systems[2].atoms.velocities[457][-2].magnitude == approx(-0.928553)  # JFR - velocities are not being read!!


def test_multiple_dump(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/2_xyz_files/log.lammps', archive, None)

    sec_systems = archive.run[0].system
    assert len(sec_systems) == 101
    assert sec_systems[2].atoms.positions[468][0].magnitude == approx(3.00831)
    assert sec_systems[-1].atoms.velocities[72][1].magnitude == approx(-4.61496)  # JFR - universe cannot be built without positions


def test_md_atomsgroup(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/polymer_melt/Emin/log.step4.0_minimization', archive, None)

    sec_run = archive.run[0]
    sec_systems = sec_run.system

    assert len(sec_systems[0].atoms_group) == 1
    assert len(sec_systems[0].atoms_group[0].atoms_group) == 100

    assert sec_systems[0].atoms_group[0].label == 'group_0'
    assert sec_systems[0].atoms_group[0].type == 'molecule_group'
    assert sec_systems[0].atoms_group[0].index == 0
    assert sec_systems[0].atoms_group[0].composition_formula == '0(100)'
    assert sec_systems[0].atoms_group[0].n_atoms == 7200
    assert sec_systems[0].atoms_group[0].atom_indices[5] == 5
    assert sec_systems[0].atoms_group[0].is_molecule is False

    assert sec_systems[0].atoms_group[0].atoms_group[52].label == '0'
    assert sec_systems[0].atoms_group[0].atoms_group[52].type == 'molecule'
    assert sec_systems[0].atoms_group[0].atoms_group[52].index == 52
    assert sec_systems[0].atoms_group[0].atoms_group[52].composition_formula == '1(1)2(1)3(1)4(1)5(1)6(1)7(1)8(1)9(1)10(1)'
    assert sec_systems[0].atoms_group[0].atoms_group[52].n_atoms == 72
    assert sec_systems[0].atoms_group[0].atoms_group[52].atom_indices[8] == 3752
    assert sec_systems[0].atoms_group[0].atoms_group[52].is_molecule is True

    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].label == 'group_8'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].type == 'monomer_group'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].index == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].composition_formula == '8(1)'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].n_atoms == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atom_indices[5] == 5527
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].is_molecule is False

    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].label == '8'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].type == 'monomer'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].index == 0
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].composition_formula == '1(4)4(2)6(1)'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].n_atoms == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].atom_indices[5] == 5527
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].is_molecule is False


# def test_rdf(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/lammps/hexane_cyclohexane/log.hexane_cyclohexane_nvt', archive, None)

#     sec_workflow = archive.workflow[0]
#     section_md = sec_workflow.molecular_dynamics.results

#     assert section_md.radial_distribution_functions[0].type == 'molecular'
#     assert section_md.radial_distribution_functions[0].n_smooth == 2
#     assert section_md.radial_distribution_functions[0].variables_name[0] == 'distance'

#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].label == '0-0'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].n_bins == 198
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].bins[122].magnitude == approx(6.923255643844605 * 10**(-10))
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].bins[122].units == 'meter'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].value[96] == approx(0.0)
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].frame_start == 0
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].frame_end == 40

#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].label == '0-0'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].n_bins == 198
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].bins[65].magnitude == approx(3.727906885147095 * 10**(-10))
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].bins[65].units == 'meter'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].value[52] == approx(0.0)
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].frame_start == 120
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[3].frame_end == 201

#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].label == '1-0'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].n_bins == 198
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].bins[102].magnitude == approx(5.802080640792847 * 10**(-10))
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].bins[102].units == 'meter'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].value[55] == approx(0.0)
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].frame_start == 40
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[5].frame_end == 201

#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].label == '1-1'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].n_bins == 198
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].bins[44].magnitude == approx(2.550673131942749 * 10**(-10))
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].bins[44].units == 'meter'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].value[101] == approx(1.4750986777470825)
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].frame_start == 80
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[10].frame_end == 201


# def test_msd(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/lammps/1_xyz_files/log.lammps', archive, None)

#     sec_workflow = archive.workflow[0]
#     section_md = sec_workflow.molecular_dynamics.results

#     assert section_md.mean_squared_displacements[0].type == 'molecular'
#     assert section_md.mean_squared_displacements[0].direction == 'xyz'

#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].label == '0'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].n_times == 42
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].times[13].magnitude == approx(13.0 * 10**(-12))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].times[13].units == 'second'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].value[32].magnitude == approx(0.4608079594680876 * 10**(-20))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].value[32].units == 'meter^2'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.value.magnitude == approx(0.002425337637745065 * 10**(-8))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.value.units == 'meter^2/second'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.error_type == 'Pearson correlation coefficient'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.errors == approx(0.9989207980765741)

#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].label == '1'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].n_times == 42
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].times[13].magnitude == approx(13.0 * 10**(-12))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].times[13].units == 'second'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].value[32].magnitude == approx(0.6809866201778795 * 10**(-20))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].value[32].units == 'meter^2'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].diffusion_constant.value.magnitude == approx(0.003761006810836386 * 10**(-8))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].diffusion_constant.value.units == 'meter^2/second'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].diffusion_constant.error_type == 'Pearson correlation coefficient'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[1].diffusion_constant.errors == approx(0.996803829564569)


def test_geometry_optimization(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/polymer_melt/Emin/log.step4.0_minimization', archive, None)

    sec_workflow = archive.workflow[0]
    section_go = sec_workflow.geometry_optimization

    assert section_go.type == 'atomic'
    assert section_go.method == 'polak_ribiere_conjugant_gradient'

    assert section_go.convergence_tolerance_energy_difference.magnitude == approx(0.0)
    assert section_go.convergence_tolerance_energy_difference.units == 'joule'
    assert section_go.final_energy_difference.magnitude == approx(0.0)
    assert section_go.final_energy_difference.units == 'joule'

    assert section_go.convergence_tolerance_force_maximum.magnitude == approx(100)
    assert section_go.convergence_tolerance_force_maximum.units == 'newton'

    assert section_go.final_force_maximum.magnitude == approx(5091750000.0)
    assert section_go.final_force_maximum.units == 'newton'

    assert section_go.optimization_steps_maximum == 10000
    assert section_go.optimization_steps == 160
    assert len(section_go.energies) == 159
    assert section_go.energies[14].magnitude == approx(6.931486093999211e-17)
    assert section_go.energies[14].units == 'joule'
    assert len(section_go.steps) == 159
    assert section_go.steps[22] == 1100


# def test_radius_of_gyration(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/lammps/polymer_melt/Equil/nohup.out', archive, None)

#     sec_calc = archive.run[0].calculation[4]
#     sec_rg = sec_calc.radius_of_gyration[0]
#     sec_rgvals = sec_rg.radius_of_gyration_values[10]

#     assert sec_rg.kind == 'molecular'
#     assert sec_rgvals.label == '0-index_10'
#     assert sec_rgvals.value.magnitude == approx(6.892062676599172e-10)
#     assert sec_rgvals.value.units == 'meter'

#     sec_calc = archive.run[0].calculation[7]
#     sec_rg = sec_calc.radius_of_gyration[0]
#     sec_rgvals = sec_rg.radius_of_gyration_values[27]

#     assert sec_rg.kind == 'molecular'
#     assert sec_rgvals.label == '0-index_27'
#     assert sec_rgvals.value.magnitude == approx(5.233325827723867e-10)
#     assert sec_rgvals.value.units == 'meter'
