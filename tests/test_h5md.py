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
    assert len(sec_run.x_h5md_version) == 2
    assert sec_run.x_h5md_version[1] == 0
    assert sec_run.x_h5md_author.name == 'Joseph F. Rudzinski'
    assert sec_run.x_h5md_author.email == 'joseph.rudzinski@physik.hu-berlin.de'
    assert sec_run.x_h5md_creator.name == 'h5py'
    assert sec_run.x_h5md_creator.version == '3.6.0'

    sec_method = sec_run.method
    sec_atom_params = sec_method[0].atom_parameters
    assert len(sec_atom_params) == 31583
    assert sec_atom_params[164].label == 'O'
    assert sec_atom_params[164].mass.magnitude == approx(2.656767806475139e-26)
    assert sec_atom_params[164].charge.magnitude == approx(-8.01088317e-20)

    assert len(sec_method[0].force_field.model[0].contributions) == 3
    assert sec_method[0].force_field.model[0].contributions[1].type == 'angles'
    assert sec_method[0].force_field.model[0].contributions[1].n_interactions == 762
    assert sec_method[0].force_field.model[0].contributions[1].n_atoms == 3
    assert sec_method[0].force_field.model[0].contributions[1].atom_labels[10][0] == 'O'
    assert sec_method[0].force_field.model[0].contributions[1].atom_indices[100][1] == 51

    assert sec_method[0].force_field.force_calculations.vdw_cutoff.magnitude == approx(1.2e-09)
    assert sec_method[0].force_field.force_calculations.vdw_cutoff.units == 'meter'
    assert sec_method[0].force_field.force_calculations.coulomb_type == 'particle_mesh_ewald'
    assert sec_method[0].force_field.force_calculations.coulomb_cutoff.magnitude == approx(1.2e-09)
    assert sec_method[0].force_field.force_calculations.coulomb_cutoff.units == 'meter'
    assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_frequency == 1
    assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.magnitude == approx(1.2e-09)
    assert sec_method[0].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.units == 'meter'

    sec_systems = sec_run.system
    assert len(sec_systems) == 5
    assert np.shape(sec_systems[0].atoms.positions) == (31583, 3)
    assert np.shape(sec_systems[0].atoms.velocities) == (31583, 3)
    assert sec_systems[0].atoms.n_atoms == 31583
    assert sec_systems[0].atoms.labels[100] == 'H'

    assert sec_systems[2].atoms.positions[800][1].magnitude == approx(2.686057472229004e-09)
    assert sec_systems[2].atoms.velocities[1200][2].magnitude == approx(40000.0)
    assert sec_systems[3].atoms.lattice_vectors[2][2].magnitude == approx(6.822318267822266e-09)
    assert sec_systems[0].atoms.bond_list[200][0] == 198

    sec_atoms_group = sec_systems[0].atoms_group
    assert len(sec_atoms_group) == 4
    assert sec_atoms_group[0].label == 'group_1ZNF'
    assert sec_atoms_group[0].type == 'molecule_group'
    assert sec_atoms_group[0].composition_formula == '1ZNF(1)'
    assert sec_atoms_group[0].n_atoms == 423
    assert sec_atoms_group[0].atom_indices[159] == 159
    assert sec_atoms_group[0].is_molecule is False
    sec_proteins = sec_atoms_group[0].atoms_group
    assert len(sec_proteins) == 1
    assert sec_proteins[0].label == '1ZNF'
    assert sec_proteins[0].type == 'molecule'
    assert sec_proteins[0].composition_formula == 'ACE(1)TYR(1)LYS(1)CYS(1)GLY(1)LEU(1)CYS(1)GLU(1)ARG(1)SER(1)PHE(1)VAL(1)GLU(1)LYS(1)SER(1)ALA(1)LEU(1)SER(1)ARG(1)HIS(1)GLN(1)ARG(1)VAL(1)HIS(1)LYS(1)ASN(1)NH2(1)'
    assert sec_proteins[0].n_atoms == 423
    assert sec_proteins[0].atom_indices[400] == 400
    assert sec_proteins[0].is_molecule is True
    sec_res_group = sec_proteins[0].atoms_group
    assert len(sec_res_group) == 27
    assert sec_res_group[14].label == 'group_ARG'
    assert sec_res_group[14].type == 'monomer_group'
    assert sec_res_group[14].composition_formula == 'ARG(1)'
    assert sec_res_group[14].n_atoms == 24
    assert sec_res_group[14].atom_indices[2] == 329
    assert sec_res_group[14].is_molecule is False
    sec_res = sec_res_group[14].atoms_group
    assert len(sec_res) == 1
    assert sec_res[0].label == 'ARG'
    assert sec_res[0].type == 'monomer'
    assert sec_res[0].composition_formula == 'C(1)CA(1)CB(1)CD(1)CG(1)CZ(1)H(1)HA(1)HB2(1)HB3(1)HD2(1)HD3(1)HE(1)HG2(1)HG3(1)HH11(1)HH12(1)HH21(1)HH22(1)N(1)NE(1)NH1(1)NH2(1)O(1)'
    assert sec_res[0].n_atoms == 24
    assert sec_res[0].atom_indices[10] == 337
    assert sec_res[0].is_molecule is False
    assert sec_res[0].x_h5md_parameters[0].kind == 'hydrophobicity'
    assert sec_res[0].x_h5md_parameters[0].value == '0.81'
    assert sec_res[0].x_h5md_parameters[0].unit is None

    sec_calc = sec_run.calculation
    assert len(sec_calc) == 5
    assert np.shape(sec_calc[1].forces.total.value) == (31583, 3)
    assert sec_calc[1].forces.total.value[2100][2].magnitude == 500.0
    assert sec_calc[2].temperature.magnitude == 300.0
    assert len(sec_calc[1].x_h5md_custom_calculations) == 1
    assert sec_calc[1].x_h5md_custom_calculations[0].kind == 'custom_thermo'
    assert sec_calc[1].x_h5md_custom_calculations[0].value == 100.0
    assert sec_calc[1].x_h5md_custom_calculations[0].unit == 'newton / angstrom ** 2'
    assert sec_calc[2].time.magnitude == 2e-12
    assert sec_calc[2].energy.kinetic.value.magnitude == approx(2000)
    assert sec_calc[2].energy.potential.value.magnitude == approx(1000)
    assert sec_calc[1].energy.x_h5md_energy_contributions[0].kind == 'energy-custom'
    assert sec_calc[1].energy.x_h5md_energy_contributions[0].value.magnitude == 3000.0

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == 'MolecularDynamics'
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == 'NPT'
    assert sec_method.integrator_type == 'langevin_leap_frog'
    assert sec_method.integration_timestep.magnitude == 2e-27
    assert sec_method.integration_timestep.units == 'second'
    assert sec_method.n_steps == 20000000
    assert sec_method.coordinate_save_frequency == 10000
    assert sec_method.thermostat_parameters[0].thermostat_type == 'langevin_leap_frog'
    assert sec_method.thermostat_parameters[0].reference_temperature.magnitude == 300.0
    assert sec_method.thermostat_parameters[0].reference_temperature.units == 'kelvin'
    assert sec_method.thermostat_parameters[0].coupling_constant.magnitude == 1e-12
    assert sec_method.thermostat_parameters[0].coupling_constant.units == 'second'
    assert sec_method.barostat_parameters[0].barostat_type == 'berendsen'
    assert sec_method.barostat_parameters[0].coupling_type == 'isotropic'
    assert np.all(sec_method.barostat_parameters[0].reference_pressure.magnitude == [[100000., 0., 0.], [0., 100000., 0.], [0., 0., 100000.]])
    assert sec_method.barostat_parameters[0].reference_pressure.units == 'pascal'
    assert np.all(sec_method.barostat_parameters[0].coupling_constant.magnitude == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert sec_method.barostat_parameters[0].coupling_constant.units == 'second'
    assert np.all(sec_method.barostat_parameters[0].compressibility.magnitude == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert sec_method.barostat_parameters[0].compressibility.units == '1 / pascal'

    sec_workflow_results = sec_workflow.results
    assert len(sec_workflow_results.ensemble_properties) == 1
    ensemble_property_0 = sec_workflow_results.ensemble_properties[0]
    assert ensemble_property_0.label == 'diffusion_constants'
    assert ensemble_property_0.error_type == 'Pearson_correlation_coefficient'
    assert len(ensemble_property_0.ensemble_property_values) == 2
    assert ensemble_property_0.ensemble_property_values[1].label == 'MOL2'
    assert ensemble_property_0.ensemble_property_values[1].errors == 0.95
    assert ensemble_property_0.ensemble_property_values[1].value_magnitude == 2.
    assert ensemble_property_0.ensemble_property_values[1].value_unit == 'angstrom ** 2 / picosecond'
    ensemble_property_1 = sec_workflow_results.radial_distribution_functions[0]
    assert ensemble_property_1.label == 'radial_distribution_functions'
    assert ensemble_property_1.type == 'molecular'
    assert len(ensemble_property_1.radial_distribution_function_values) == 3
    assert ensemble_property_1.radial_distribution_function_values[1].label == 'MOL1-MOL2'
    assert ensemble_property_1.radial_distribution_function_values[1].n_bins == 651
    assert ensemble_property_1.radial_distribution_function_values[1].frame_start == 0
    assert ensemble_property_1.radial_distribution_function_values[1].frame_end == 4
    assert ensemble_property_1.radial_distribution_function_values[1].bins[51].magnitude == approx(2.55e-11)
 #   assert ensemble_property_1.radial_distribution_function_values[1].bins_unit == 'angstrom'
    assert ensemble_property_1.radial_distribution_function_values[1].value[51] == approx(0.284764)
    correlation_function_0 = sec_workflow_results.mean_squared_displacements[0]
    assert correlation_function_0.type == 'molecular'
    assert correlation_function_0.label == 'mean_squared_displacements'
    assert correlation_function_0.direction == 'xyz'
    assert correlation_function_0.error_type == 'standard_deviation'
    assert len(correlation_function_0.mean_squared_displacement_values) == 2
    assert correlation_function_0.mean_squared_displacement_values[0].label == 'MOL1'
    assert correlation_function_0.mean_squared_displacement_values[0].n_times == 51
    assert correlation_function_0.mean_squared_displacement_values[0].times[10].magnitude == approx(2.e-11)
    assert correlation_function_0.mean_squared_displacement_values[0].value[10].magnitude == approx(6.79723e-21)
#    assert correlation_function_0.mean_squared_displacement_values[0].value[10].units == 'angstrom ** 2'
    assert correlation_function_0.mean_squared_displacement_values[0].errors[10] == approx(0.0)