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
    assert sec_workflow.type == 'molecular_dynamics'
    assert sec_workflow.molecular_dynamics.x_lammps_integrator_dt.magnitude == 2.5e-16
    assert sec_workflow.molecular_dynamics.x_lammps_thermostat_target_temperature.magnitude == 300.
    assert sec_workflow.molecular_dynamics.ensemble_type == 'NVT'

    sec_method = sec_run.method[0]
    assert len(sec_method.force_field.model[0].contributions) == 4
    assert sec_method.force_field.model[0].contributions[2].type == 'harmonic'
    assert sec_method.force_field.model[0].contributions[0].parameters[0][2] == 0.066

    sec_system = sec_run.system
    assert len(sec_system) == 201
    assert sec_system[5].atoms.lattice_vectors[1][1].magnitude == approx(2.24235e-09)
    assert False not in sec_system[0].atoms.periodic
    assert sec_system[80].atoms.labels[91:96] == ['H', 'H', 'H', 'C', 'C']

    sec_scc = sec_run.calculation
    assert len(sec_scc) == 201
    assert sec_scc[21].energy.current.value.magnitude == approx(8.86689197e-18)
    assert sec_scc[180].time_calculation.magnitude == 218.5357
    assert sec_scc[56].thermodynamics[0].pressure.magnitude == approx(-77642135.4975)
    assert sec_scc[103].thermodynamics[0].temperature.magnitude == 291.4591
    assert sec_scc[11].thermodynamics[0].time_step == 4400
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
    parser.parse('tests/data/lammps/polymer_melt/log.step4.0_minimization', archive, None)

    sec_run = archive.run[0]
    sec_systems = sec_run.system

    assert len(sec_systems[0].atoms_group) == 1
    assert len(sec_systems[0].atoms_group[0].atoms_group) == 100

    assert sec_systems[0].atoms_group[0].label == 'seg_0_0'
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

    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].label == '8'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].type == 'monomer'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].index == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].composition_formula == '1(4)4(2)6(1)'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].n_atoms == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atom_indices[5] == 5527
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].is_molecule is False


def test_rdf(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/hexane_cyclohexane/log.hexane_cyclohexane_nvt', archive, None)

    sec_workflow = archive.workflow[0]
    section_MD = sec_workflow.molecular_dynamics

    assert section_MD.radial_distribution_functions[0].label == 'molecular radial distribution functions'
    assert section_MD.radial_distribution_functions[0].n_smooth == 2
    assert section_MD.radial_distribution_functions[0].variables_name == 'distance'

    assert section_MD.radial_distribution_functions[0].rdf_values[0].type == '0-0'
    assert section_MD.radial_distribution_functions[0].rdf_values[0].bins[0][122].magnitude == approx(6.9232556438446045 * 10**(-10))
    assert section_MD.radial_distribution_functions[0].rdf_values[0].bins[0][122].units == 'meter'
    assert section_MD.radial_distribution_functions[0].rdf_values[0].value[96] == approx(0.5017477701631716)

    assert section_MD.radial_distribution_functions[0].rdf_values[1].type == '1-0'
    assert section_MD.radial_distribution_functions[0].rdf_values[1].bins[0][102].magnitude == approx(5.8020806407928465 * 10**(-10))
    assert section_MD.radial_distribution_functions[0].rdf_values[1].bins[0][102].units == 'meter'
    assert section_MD.radial_distribution_functions[0].rdf_values[1].value[55] == approx(0.0)


def test_msd(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/1_xyz_files/log.lammps', archive, None)

    sec_workflow = archive.workflow[0]
    section_MD = sec_workflow.molecular_dynamics

    assert section_MD.mean_squared_displacements[0].label == 'molecular mean squared displacements'

    assert section_MD.mean_squared_displacements[0].msd_values[0].type == '0'
    assert section_MD.mean_squared_displacements[0].msd_values[0].times[13].magnitude == approx(13.0 * 10**(-12))
    assert section_MD.mean_squared_displacements[0].msd_values[0].times[13].units == 'second'
    assert section_MD.mean_squared_displacements[0].msd_values[0].value[32].magnitude == approx(0.4608079594680876 * 10**(-20))
    assert section_MD.mean_squared_displacements[0].msd_values[0].value[32].units == 'meter^2'
    assert section_MD.mean_squared_displacements[0].msd_values[0].diffusion_constant.value.magnitude == approx(0.002425337637745065 * 10**(-8))
    assert section_MD.mean_squared_displacements[0].msd_values[0].diffusion_constant.value.units == 'meter^2/second'
    assert section_MD.mean_squared_displacements[0].msd_values[0].diffusion_constant.error_type == 'Pearson correlation coefficient'
    assert section_MD.mean_squared_displacements[0].msd_values[0].diffusion_constant.error_value.magnitude == approx(0.9989207980765741 * 10**(-8))
    assert section_MD.mean_squared_displacements[0].msd_values[0].diffusion_constant.error_value.units == 'meter^2/second'

    assert section_MD.mean_squared_displacements[0].msd_values[1].type == '1'
    assert section_MD.mean_squared_displacements[0].msd_values[1].times[13].magnitude == approx(13.0 * 10**(-12))
    assert section_MD.mean_squared_displacements[0].msd_values[1].times[13].units == 'second'
    assert section_MD.mean_squared_displacements[0].msd_values[1].value[32].magnitude == approx(0.6809866201778795 * 10**(-20))
    assert section_MD.mean_squared_displacements[0].msd_values[1].value[32].units == 'meter^2'
    assert section_MD.mean_squared_displacements[0].msd_values[1].diffusion_constant.value.magnitude == approx(0.003761006810836386 * 10**(-8))
    assert section_MD.mean_squared_displacements[0].msd_values[1].diffusion_constant.value.units == 'meter^2/second'
    assert section_MD.mean_squared_displacements[0].msd_values[1].diffusion_constant.error_type == 'Pearson correlation coefficient'
    assert section_MD.mean_squared_displacements[0].msd_values[1].diffusion_constant.error_value.magnitude == approx(0.996803829564569 * 10**(-8))
    assert section_MD.mean_squared_displacements[0].msd_values[1].diffusion_constant.error_value.units == 'meter^2/second'
