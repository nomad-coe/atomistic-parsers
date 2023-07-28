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
from atomisticparsers.h5md import H5MDParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return H5MDParser()


def test_md(parser):
    archive = EntryArchive()
    parser.parse('tests/data/h5md/openmm/test_traj_openmm_5frames.h5', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.name == 'OpenMM'
    assert sec_run.program.version == '-1.-1.-1'

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == 'MolecularDynamics'
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == 'NPT'
    assert sec_method.x_h5md_integrator_type == 'langevin_leap_frog'
    assert sec_method.integration_timestep.magnitude == 2e-27
    assert sec_method.integration_timestep.units == 'second'
    assert sec_method.n_steps == 20000000
    assert sec_method.coordinate_save_frequency == 10000
    assert sec_method.thermostat_parameters.x_h5md_thermostat_type == 'langevin_leap_frog'
    assert sec_method.thermostat_parameters.reference_temperature.magnitude == 300.0
    assert sec_method.thermostat_parameters.reference_temperature.units == 'kelvin'
    assert sec_method.thermostat_parameters.coupling_constant.magnitude == 1e-12
    assert sec_method.thermostat_parameters.coupling_constant.units == 'second'
    assert sec_method.barostat_parameters.barostat_type == 'berendsen'
    assert sec_method.barostat_parameters.coupling_type == 'isotropic'
    assert np.all(sec_method.barostat_parameters.reference_pressure.magnitude == [[100000., 0., 0.], [0., 100000., 0.], [0., 0., 100000.]])
    assert sec_method.barostat_parameters.reference_pressure.units == 'pascal'
    assert np.all(sec_method.barostat_parameters.coupling_constant.magnitude == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert sec_method.barostat_parameters.coupling_constant.units == 'second'
    assert np.all(sec_method.barostat_parameters.compressibility.magnitude == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert sec_method.barostat_parameters.compressibility.units == '1 / pascal'

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 5
    assert sec_calc[2].temperature.magnitude == 300.0
    assert sec_calc[2].time.magnitude  == 2.0
    assert sec_calc[2].energy.kinetic.value.magnitude == approx(1000)
    assert sec_calc[2].energy.potential.value.magnitude == approx(1000)

    sec_systems = sec_run.system
    assert len(sec_systems) == 5
    assert np.shape(sec_systems[0].atoms.positions) == (31583, 3)
    assert sec_systems[0].atoms.n_atoms = 31583
    assert sec_systems[0].atoms.labels[100] = 'H'

    assert sec_systems[2].atoms.positions[800][1].magnitude == approx(2.686057472229004e-09)
    assert sec_systems[3].atoms.lattice_vectors[2][2].magnitude == approx(6.822318267822266e-09)
    assert sec_systems[0].atoms.bond_list[200][0] == 198

    # sec_method = sec_run.method
    # assert len(sec_method) == 1
    # assert len(sec_method[0].force_field.model[0].contributions) == 8
    # assert sec_method[0].force_field.model[0].contributions[6].type == 'bond'
    # assert sec_method[0].force_field.model[0].contributions[6].n_inter == 1017
    # assert sec_method[0].force_field.model[0].contributions[6].n_atoms == 2
    # assert sec_method[0].force_field.model[0].contributions[6].atom_labels[10][0] == 'C'
    # assert sec_method[0].force_field.model[0].contributions[6].atom_indices[100][1] == 141
    # assert sec_method[0].force_field.model[0].contributions[6].parameters[858] == approx(0.9999996193044006)

    # assert sec_method[0].force_field.force_calculations.vdw_cutoff.magnitude == 1.2e-09
    # assert sec_method[0].force_field.force_calculations.vdw_cutoff.units == 'meter'
    # assert sec_method[0].force_field.force_calculations.coulomb_type == 'particle_mesh_ewald'
    # assert sec_method[0].force_field.force_calculations.coulomb_cutoff.magnitude == 0.9
    # assert sec_method[0].force_field.force_calculations.coulomb_cutoff.units == 'meter'
    # assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_frequency == 5
    # assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.magnitude == 9.000000000000001e-10
    # assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.units == 'meter'


# def test_md_edr(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/gromacs/fe_test/mdrun.out', archive, None)

#     assert len(archive.run[0].calculation) == 5


# def test_md_atomsgroup(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/gromacs/polymer_melt/step4.0_minimization.log', archive, None)

#     sec_run = archive.run[0]
#     sec_systems = sec_run.system

#     assert len(sec_systems[0].atoms_group) == 1
#     assert len(sec_systems[0].atoms_group[0].atoms_group) == 100

#     assert sec_systems[0].atoms_group[0].label == 'group_S1P1'
#     assert sec_systems[0].atoms_group[0].type == 'molecule_group'
#     assert sec_systems[0].atoms_group[0].index == 0
#     assert sec_systems[0].atoms_group[0].composition_formula == 'S1P1(100)'
#     assert sec_systems[0].atoms_group[0].n_atoms == 7200
#     assert sec_systems[0].atoms_group[0].atom_indices[5] == 5
#     assert sec_systems[0].atoms_group[0].is_molecule is False

#     assert sec_systems[0].atoms_group[0].atoms_group[52].label == 'S1P1'
#     assert sec_systems[0].atoms_group[0].atoms_group[52].type == 'molecule'
#     assert sec_systems[0].atoms_group[0].atoms_group[52].index == 52
#     assert sec_systems[0].atoms_group[0].atoms_group[52].composition_formula == 'ETHOX(10)'
#     assert sec_systems[0].atoms_group[0].atoms_group[52].n_atoms == 72
#     assert sec_systems[0].atoms_group[0].atoms_group[52].atom_indices[8] == 3752
#     assert sec_systems[0].atoms_group[0].atoms_group[52].is_molecule is True

#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].label == 'group_ETHOX'
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].type == 'monomer_group'
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].index == 0
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].composition_formula == 'ETHOX(10)'
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].n_atoms == 72
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atom_indices[5] == 5477
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].is_molecule is False

#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].label == 'ETHOX'
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].type == 'monomer'
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].index == 7
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].composition_formula == 'C(2)H(4)O(1)'
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].n_atoms == 7
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].atom_indices[5] == 5527
#     assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].is_molecule is False


# def test_rdf(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/gromacs/fe_test/mdrun.out', archive, None)

#     sec_workflow = archive.workflow2
#     section_md = sec_workflow.results

#     assert section_md.radial_distribution_functions[0].type == 'molecular'
#     assert section_md.radial_distribution_functions[0].n_smooth == 2
#     assert section_md.radial_distribution_functions[0].variables_name[0] == 'distance'

#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].label == 'SOL-Protein'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].n_bins == 198
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].bins[122].magnitude == approx(7.624056451320648 * 10**(-10))
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].bins[122].units == 'meter'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].value[96] == approx(1.093694948374587)
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].frame_start == 0
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[0].frame_end == 2

#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].label == 'SOL-SOL'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].n_bins == 198
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].bins[102].magnitude == approx(6.389391438961029 * 10**(-10))
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].bins[102].units == 'meter'
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].value[55] == approx(0.8368052672121375)
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].frame_start == 0
#     assert section_md.radial_distribution_functions[0].radial_distribution_function_values[1].frame_end == 2


# def test_msd(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/gromacs/cgwater/mdrun.log', archive, None)

#     sec_workflow = archive.workflow2
#     section_md = sec_workflow.results

#     assert section_md.mean_squared_displacements[0].type == 'molecular'
#     assert section_md.mean_squared_displacements[0].direction == 'xyz'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].label == 'LJ'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].n_times == 54
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].times[52].magnitude == approx(95.0 * 10**(-12))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].times[52].units == 'second'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].value[32].magnitude == approx(250.15309179080856 * 10**(-20))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].value[32].units == 'meter^2'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.value.magnitude == approx(1.1311880364159048 * 10**(-8))
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.value.units == 'meter^2/second'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.error_type == 'Pearson correlation coefficient'
#     assert section_md.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.errors == 0.9999312519176002


# def test_geometry_optimization(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/gromacs/polymer_melt/step4.0_minimization.log', archive, None)

#     sec_workflow = archive.workflow2

#     assert sec_workflow.method.type == 'atomic'
#     assert sec_workflow.method.method == 'steepest_descent'
#     assert sec_workflow.method.convergence_tolerance_force_maximum.magnitude == approx(6.02214076e+38)
#     assert sec_workflow.method.convergence_tolerance_force_maximum.units == 'newton'
#     assert sec_workflow.results.final_force_maximum.magnitude == approx(1.303670442204273e+38)
#     assert sec_workflow.results.final_force_maximum.units == 'newton'
#     assert sec_workflow.results.optimization_steps == 12
#     assert sec_workflow.method.optimization_steps_maximum == 5000
#     assert len(sec_workflow.results.energies) == 11
#     assert sec_workflow.results.energies[2].magnitude == approx(8.244726173423345e-17)
#     assert sec_workflow.results.energies[2].units == 'joule'
#     assert len(sec_workflow.results.steps) == 11
#     assert sec_workflow.results.steps[4] == 5000
