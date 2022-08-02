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
from atomisticparsers.gromacs import GromacsParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return GromacsParser()


def test_md_verbose(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gromacs/fe_test/md.log', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '5.1.4'
    sec_control = sec_run.x_gromacs_section_control_parameters
    assert sec_control.x_gromacs_inout_control_coulombtype == 'PME'
    assert np.shape(sec_control.x_gromacs_inout_control_deform) == (3, 3)

    sec_workflow = archive.workflow[0]
    section_MD = sec_workflow.molecular_dynamics
    assert sec_workflow.type == 'molecular_dynamics'
    assert section_MD.thermodynamic_ensemble == 'NPT'
    assert section_MD.finished_normally is None
    assert section_MD.with_trajectory is None
    assert section_MD.with_thermodynamics is None
    assert section_MD.integration_parameters.integrator_type == 'leap_frog'
    assert section_MD.integration_parameters.integration_timestep.magnitude == 5e-16
    assert section_MD.integration_parameters.integration_timestep.units == 'second'
    assert section_MD.integration_parameters.n_steps == 20
    assert section_MD.integration_parameters.coordinate_save_frequency == 20
    assert section_MD.integration_parameters.thermodynamics_save_frequency == 5
    assert section_MD.integration_parameters.thermostat_parameters.thermostat_type == 'berendsen'
    assert section_MD.integration_parameters.thermostat_parameters.reference_temperature.magnitude == 298.0
    assert section_MD.integration_parameters.thermostat_parameters.reference_temperature.units == 'kelvin'
    assert section_MD.integration_parameters.thermostat_parameters.coupling_constant.magnitude == 5e-13
    assert section_MD.integration_parameters.thermostat_parameters.coupling_constant.units == 'second'
    assert section_MD.integration_parameters.barostat_parameters.barostat_type == 'berendsen'
    assert section_MD.integration_parameters.barostat_parameters.coupling_type == 'isotropic'
    assert np.all(section_MD.integration_parameters.barostat_parameters.reference_pressure.magnitude == [[100000., 0., 0.], [0., 100000., 0.], [0., 0., 100000.]])
    assert section_MD.integration_parameters.barostat_parameters.reference_pressure.units == 'pascal'
    assert np.all(section_MD.integration_parameters.barostat_parameters.coupling_constant.magnitude == [[1.e-12, 1.e-12, 1.e-12], [1.e-12, 1.e-12, 1.e-12], [1.e-12, 1.e-12, 1.e-12]])
    assert section_MD.integration_parameters.barostat_parameters.coupling_constant.units == 'second'
    assert np.all(section_MD.integration_parameters.barostat_parameters.compressibility.magnitude == [[4.6e-10, 0.0e+00, 0.0e+00], [0.0e+00, 4.6e-10, 0.0e+00], [0.0e+00, 0.0e+00, 4.6e-10]])
    assert section_MD.integration_parameters.barostat_parameters.compressibility.units == '1 / pascal'

    sec_sccs = sec_run.calculation
    assert len(sec_sccs) == 7
    assert sec_sccs[2].energy.total.value.magnitude == approx(-3.2711290665182795e-17)
    assert sec_sccs[5].pressure.magnitude == approx(-63926916.5)
    assert sec_sccs[-2].energy.contributions[1].value.magnitude == approx(-4.15778738e-17)
    assert sec_sccs[0].forces.total.value[5][2].magnitude == approx(-7.932968909721231e-10)

    sec_systems = sec_run.system
    assert len(sec_systems) == 2
    assert np.shape(sec_systems[0].atoms.positions) == (1516, 3)
    assert sec_systems[1].atoms.positions[800][1].magnitude == approx(2.4740036e-09)
    assert sec_systems[0].atoms.velocities[500][0].magnitude == approx(869.4773)
    assert sec_systems[1].atoms.lattice_vectors[2][2].magnitude == approx(2.469158e-09)

    sec_method = sec_run.method
    assert len(sec_method) == 1
    assert len(sec_method[0].force_field.model[0].contributions) == 1127
    assert sec_method[0].force_field.model[0].contributions[0].type == 'angle'
    assert sec_method[0].force_field.model[0].contributions[1120].parameters[1] == 575.0
    assert sec_method[0].force_field.force_calculations.vdw_cutoff.magnitude == 1.2e-09
    assert sec_method[0].force_field.force_calculations.vdw_cutoff.units == 'meter'
    assert sec_method[0].force_field.force_calculations.Coulomb_type == 'particle_mesh_ewald'
    assert sec_method[0].force_field.force_calculations.Coulomb_cutoff.magnitude == 0.9
    assert sec_method[0].force_field.force_calculations.Coulomb_cutoff.units == 'meter'
    assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_frequency == 5
    assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.magnitude == 9.000000000000001e-10
    assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.units == 'meter'


def test_md_edr(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gromacs/fe_test/mdrun.out', archive, None)

    assert len(archive.run[0].calculation) == 7


def test_md_atomsgroup(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gromacs/polymer_melt/step4.0_minimization.log', archive, None)

    sec_run = archive.run[0]
    sec_systems = sec_run.system

    assert len(sec_systems[0].atoms_group) == 1
    assert len(sec_systems[0].atoms_group[0].atoms_group) == 100

    assert sec_systems[0].atoms_group[0].label == 'group_S1P1'
    assert sec_systems[0].atoms_group[0].type == 'molecule_group'
    assert sec_systems[0].atoms_group[0].index == 0
    assert sec_systems[0].atoms_group[0].composition_formula == 'S1P1(100)'
    assert sec_systems[0].atoms_group[0].n_atoms == 7200
    assert sec_systems[0].atoms_group[0].atom_indices[5] == 5
    assert sec_systems[0].atoms_group[0].is_molecule is False

    assert sec_systems[0].atoms_group[0].atoms_group[52].label == 'S1P1'
    assert sec_systems[0].atoms_group[0].atoms_group[52].type == 'molecule'
    assert sec_systems[0].atoms_group[0].atoms_group[52].index == 52
    assert sec_systems[0].atoms_group[0].atoms_group[52].composition_formula == 'ETHOX(10)'
    assert sec_systems[0].atoms_group[0].atoms_group[52].n_atoms == 72
    assert sec_systems[0].atoms_group[0].atoms_group[52].atom_indices[8] == 3752
    assert sec_systems[0].atoms_group[0].atoms_group[52].is_molecule is True

    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].label == 'group_ETHOX'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].type == 'monomer_group'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].index == 0
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].composition_formula == 'ETHOX(10)'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].n_atoms == 72
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atom_indices[5] == 5477
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].is_molecule is False

    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].label == 'ETHOX'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].type == 'monomer'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].index == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].composition_formula == 'C(2)H(4)O(1)'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].n_atoms == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].atom_indices[5] == 5527
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].is_molecule is False


def test_rdf(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gromacs/fe_test/mdrun.out', archive, None)

    sec_workflow = archive.workflow[0]
    section_MD = sec_workflow.molecular_dynamics.results

    assert section_MD.radial_distribution_functions[0].type == 'molecular'
    assert section_MD.radial_distribution_functions[0].n_smooth == 2
    assert section_MD.radial_distribution_functions[0].variables_name[0] == 'distance'

    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[0].label == 'SOL-Protein'
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[0].n_bins == 198
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[0].bins[122].magnitude == approx(7.624056451320648 * 10**(-10))
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[0].bins[122].units == 'meter'
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[0].value[96] == approx(1.093694948374587)

    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[1].label == 'SOL-SOL'
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[1].n_bins == 198
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[1].bins[102].magnitude == approx(6.389391438961029 * 10**(-10))
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[1].bins[102].units == 'meter'
    assert section_MD.radial_distribution_functions[0].radial_distribution_function_values[1].value[55] == approx(0.8368052672121375)


def test_msd(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gromacs/cgwater/mdrun.log', archive, None)

    sec_workflow = archive.workflow[0]
    section_MD = sec_workflow.molecular_dynamics.results

    assert section_MD.mean_squared_displacements[0].type == 'molecular'
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].label == 'LJ'
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].n_times == 54
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].times[52].magnitude == approx(95.0 * 10**(-12))
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].times[52].units == 'second'
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].value[32].magnitude == approx(250.15309179080856 * 10**(-20))
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].value[32].units == 'meter^2'
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.value.magnitude == approx(1.1311880364159048 * 10**(-8))
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.value.units == 'meter^2/second'
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.error_type == 'Pearson correlation coefficient'
    assert section_MD.mean_squared_displacements[0].mean_squared_displacement_values[0].diffusion_constant.errors == 0.9999312519176002
