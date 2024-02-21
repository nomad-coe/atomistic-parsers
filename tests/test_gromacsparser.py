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


@pytest.fixture(scope="module")
def parser():
    return GromacsParser()


def test_md_verbose(parser):
    archive = EntryArchive()
    parser.parse("tests/data/gromacs/fe_test/md.log", archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == "5.1.4"
    sec_control = sec_run.x_gromacs_section_control_parameters
    assert sec_control.x_gromacs_inout_control_coulombtype == "pme"
    assert np.shape(sec_control.x_gromacs_inout_control_deform) == (3, 3)

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == "MolecularDynamics"
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == "NPT"
    assert sec_method.integrator_type == "leap_frog"
    assert sec_method.integration_timestep.magnitude == 5e-16
    assert sec_method.integration_timestep.units == "second"
    assert sec_method.n_steps == 20
    assert sec_method.coordinate_save_frequency == 20
    assert sec_method.thermodynamics_save_frequency == 5
    assert sec_method.thermostat_parameters[0].thermostat_type == "berendsen"
    assert sec_method.thermostat_parameters[0].reference_temperature.magnitude == 298.0
    assert sec_method.thermostat_parameters[0].reference_temperature.units == "kelvin"
    assert sec_method.thermostat_parameters[0].coupling_constant.magnitude == 5e-13
    assert sec_method.thermostat_parameters[0].coupling_constant.units == "second"
    assert sec_method.barostat_parameters[0].barostat_type == "berendsen"
    assert sec_method.barostat_parameters[0].coupling_type == "isotropic"
    assert np.all(
        sec_method.barostat_parameters[0].reference_pressure.magnitude
        == [[100000.0, 0.0, 0.0], [0.0, 100000.0, 0.0], [0.0, 0.0, 100000.0]]
    )
    assert sec_method.barostat_parameters[0].reference_pressure.units == "pascal"
    assert np.all(
        sec_method.barostat_parameters[0].coupling_constant.magnitude
        == [
            [1.0e-12, 1.0e-12, 1.0e-12],
            [1.0e-12, 1.0e-12, 1.0e-12],
            [1.0e-12, 1.0e-12, 1.0e-12],
        ]
    )
    assert sec_method.barostat_parameters[0].coupling_constant.units == "second"
    assert np.all(
        sec_method.barostat_parameters[0].compressibility.magnitude
        == [
            [4.6e-10, 0.0e00, 0.0e00],
            [0.0e00, 4.6e-10, 0.0e00],
            [0.0e00, 0.0e00, 4.6e-10],
        ]
    )
    assert sec_method.barostat_parameters[0].compressibility.units == "1 / pascal"

    sec_sccs = sec_run.calculation
    assert len(sec_sccs) == 5
    assert sec_sccs[1].pressure_tensor[1][2].magnitude == approx(40267181.396484375)
    assert sec_sccs[3].pressure.magnitude == approx(-63926916.50390625)
    assert sec_sccs[3].temperature.magnitude == approx(291.80401611328125)
    assert sec_sccs[2].volume.magnitude == approx(1.505580043792725e-26)
    assert sec_sccs[2].density.magnitude == approx(1007.9478759765625)
    assert sec_sccs[2].enthalpy.magnitude == approx(-1.184108268425108e31)
    assert sec_sccs[2].virial_tensor[2][2].magnitude == approx(1.1367756347656254e-19)
    assert len(sec_sccs[1].x_gromacs_thermodynamics_contributions) == 5
    assert sec_sccs[1].x_gromacs_thermodynamics_contributions[2].kind == "#Surf*SurfTen"
    assert sec_sccs[1].x_gromacs_thermodynamics_contributions[2].value == approx(
        2453.242431640625
    )
    assert len(sec_sccs[4].energy.x_gromacs_energy_contributions) == 12
    assert sec_sccs[-2].energy.x_gromacs_energy_contributions[1].kind == "G96Angle"
    assert sec_sccs[-2].energy.x_gromacs_energy_contributions[
        1
    ].value.magnitude == approx(9.90594089232063e27)
    assert sec_sccs[0].energy.total.value.magnitude == approx(-1.1863129365544755e31)
    assert sec_sccs[0].energy.electrostatic.value.magnitude == approx(
        -1.6677869795296e31
    )
    assert sec_sccs[0].energy.electrostatic.short_range.magnitude == approx(
        -1.5069901728906464e31
    )
    assert sec_sccs[0].energy.electrostatic.long_range.magnitude == approx(
        -1.6079680663895344e30
    )
    assert sec_sccs[-1].energy.van_der_waals.value.magnitude == approx(
        2.5995702480888255e30
    )
    assert sec_sccs[-1].energy.van_der_waals.short_range.magnitude == approx(
        2.675488981642447e30
    )
    assert sec_sccs[-1].energy.van_der_waals.long_range.magnitude == approx(
        -4.4191382265877185e28
    )
    assert sec_sccs[-1].energy.van_der_waals.correction.magnitude == approx(
        -3.172735128774431e28
    )
    assert sec_sccs[0].energy.pressure_volume_work.value.magnitude == approx(
        5.46058641332406e26
    )

    assert sec_sccs[0].forces.total.value[5][2].magnitude == approx(
        -7.932968909721231e-10
    )

    sec_systems = sec_run.system
    assert len(sec_systems) == 2
    assert np.shape(sec_systems[0].atoms.positions) == (1516, 3)
    assert sec_systems[1].atoms.positions[800][1].magnitude == approx(2.4740036e-09)
    assert sec_systems[0].atoms.velocities[500][0].magnitude == approx(869.4773)
    assert sec_systems[1].atoms.lattice_vectors[2][2].magnitude == approx(2.469158e-09)
    assert sec_systems[0].atoms.bond_list[200][0] == 289

    sec_method = sec_run.method
    assert len(sec_method) == 1
    assert len(sec_method[0].force_field.model[0].contributions) == 8
    assert sec_method[0].force_field.model[0].contributions[6].type == "bond"
    assert sec_method[0].force_field.model[0].contributions[6].n_interactions == 1017
    assert sec_method[0].force_field.model[0].contributions[6].n_atoms == 2
    assert sec_method[0].force_field.model[0].contributions[6].atom_labels[10][0] == "C"
    assert (
        sec_method[0].force_field.model[0].contributions[6].atom_indices[100][1] == 141
    )
    assert sec_method[0].force_field.model[0].contributions[6].parameters[
        858
    ] == approx(0.9999996193044006)
    assert sec_method[0].force_field.force_calculations.vdw_cutoff.magnitude == 1.2e-09
    assert sec_method[0].force_field.force_calculations.vdw_cutoff.units == "meter"
    assert (
        sec_method[0].force_field.force_calculations.coulomb_type
        == "particle_mesh_ewald"
    )
    assert sec_method[0].force_field.force_calculations.coulomb_cutoff.magnitude == 0.9
    assert sec_method[0].force_field.force_calculations.coulomb_cutoff.units == "meter"
    assert (
        sec_method[
            0
        ].force_field.force_calculations.neighbor_searching.neighbor_update_frequency
        == 5
    )
    assert (
        sec_method[
            0
        ].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.magnitude
        == 9.000000000000001e-10
    )
    assert (
        sec_method[
            0
        ].force_field.force_calculations.neighbor_searching.neighbor_update_cutoff.units
        == "meter"
    )


def test_md_edr(parser):
    archive = EntryArchive()
    parser.parse("tests/data/gromacs/fe_test/mdrun.out", archive, None)

    assert len(archive.run[0].calculation) == 5


def test_md_atomsgroup(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/polymer_melt/step4.0_minimization.log", archive, None
    )

    sec_run = archive.run[0]
    sec_systems = sec_run.system

    assert len(sec_systems[0].atoms_group) == 1
    assert len(sec_systems[0].atoms_group[0].atoms_group) == 100

    assert sec_systems[0].atoms_group[0].label == "group_S1P1"
    assert sec_systems[0].atoms_group[0].type == "molecule_group"
    assert sec_systems[0].atoms_group[0].index == 0
    assert sec_systems[0].atoms_group[0].composition_formula == "S1P1(100)"
    assert sec_systems[0].atoms_group[0].n_atoms == 7200
    assert sec_systems[0].atoms_group[0].atom_indices[5] == 5
    assert sec_systems[0].atoms_group[0].is_molecule is False

    assert sec_systems[0].atoms_group[0].atoms_group[52].label == "S1P1"
    assert sec_systems[0].atoms_group[0].atoms_group[52].type == "molecule"
    assert sec_systems[0].atoms_group[0].atoms_group[52].index == 52
    assert (
        sec_systems[0].atoms_group[0].atoms_group[52].composition_formula == "ETHOX(10)"
    )
    assert sec_systems[0].atoms_group[0].atoms_group[52].n_atoms == 72
    assert sec_systems[0].atoms_group[0].atoms_group[52].atom_indices[8] == 3752
    assert sec_systems[0].atoms_group[0].atoms_group[52].is_molecule is True

    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].label
        == "group_ETHOX"
    )
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].type
        == "monomer_group"
    )
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].index == 0
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].composition_formula
        == "ETHOX(10)"
    )
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].n_atoms == 72
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atom_indices[5]
        == 5477
    )
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].is_molecule
        is False
    )

    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[0]
        .atoms_group[7]
        .label
        == "ETHOX"
    )
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[0].atoms_group[7].type
        == "monomer"
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[0]
        .atoms_group[7]
        .index
        == 7
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[0]
        .atoms_group[7]
        .composition_formula
        == "C(2)H(4)O(1)"
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[0]
        .atoms_group[7]
        .n_atoms
        == 7
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[0]
        .atoms_group[7]
        .atom_indices[5]
        == 5527
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[0]
        .atoms_group[7]
        .is_molecule
        is False
    )


def test_geometry_optimization(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/polymer_melt/step4.0_minimization.log", archive, None
    )

    sec_workflow = archive.workflow2

    assert sec_workflow.method.type == "atomic"
    assert sec_workflow.method.method == "steepest_descent"
    assert sec_workflow.method.convergence_tolerance_force_maximum.magnitude == approx(
        6.02214076e38
    )
    assert sec_workflow.method.convergence_tolerance_force_maximum.units == "newton"
    assert sec_workflow.results.final_force_maximum.magnitude == approx(
        1.303670442204273e38
    )
    assert sec_workflow.results.final_force_maximum.units == "newton"
    assert sec_workflow.results.optimization_steps == 12
    assert sec_workflow.method.optimization_steps_maximum == 5000
    assert len(sec_workflow.results.energies) == 11
    assert sec_workflow.results.energies[2].magnitude == approx(2.9900472759121395e31)
    assert sec_workflow.results.energies[2].units == "joule"
    assert len(sec_workflow.results.steps) == 11
    assert sec_workflow.results.steps[4] == 5000


def test_integrator_sd(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/water_AA_ENUM_tests/integrator-sd/md.log", archive, None
    )

    sec_run = archive.run[0]
    # assert sec_run.program.version == "2018.6"

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == "MolecularDynamics"
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == "NVT"
    assert sec_method.integrator_type == "langevin_goga"
    assert sec_method.thermostat_parameters[0].thermostat_type == "langevin_goga"
    assert sec_method.thermostat_parameters[0].reference_temperature.magnitude == 298.0
    assert sec_method.thermostat_parameters[0].coupling_constant.magnitude == 5e-13


def test_integrator_mdvv(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/water_AA_ENUM_tests/integrator-mdvv/md.log", archive, None
    )

    sec_run = archive.run[0]
    # assert sec_run.program.version == "2018.6"

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == "MolecularDynamics"
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == "NVE"
    assert sec_method.integrator_type == "velocity_verlet"


def test_integrator_bd(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/water_AA_ENUM_tests/integrator-bd/md.log", archive, None
    )

    sec_run = archive.run[0]
    # assert sec_run.program.version == "2018.6"

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == "MolecularDynamics"
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == "NVE"
    assert sec_method.integrator_type == "brownian"


# TODO test for andersen thermostat? It's not clear how to run this at the moment or if it is deprecated in newer versions of Gromacs.


def test_integrator_md_thermostat_vrescale(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/water_AA_ENUM_tests/integrator-md/thermostat-vrescale/md.log",
        archive,
        None,
    )

    sec_run = archive.run[0]
    assert sec_run.program.version == "2018.6"

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == "MolecularDynamics"
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == "NVT"
    assert sec_method.integrator_type == "leap_frog"
    assert sec_method.thermostat_parameters[0].thermostat_type == "velocity_rescaling"
    assert sec_method.thermostat_parameters[0].reference_temperature.magnitude == 298.0
    assert sec_method.thermostat_parameters[0].coupling_constant.magnitude == 5e-13


def test_integrator_md_thermostat_nosehoover_barostat_parrinellorahman(parser):
    archive = EntryArchive()
    parser.parse(
        "tests/data/gromacs/water_AA_ENUM_tests/integrator-md/thermostat-nosehoover_barostat-parrinellorahman/md.log",
        archive,
        None,
    )

    sec_run = archive.run[0]
    assert sec_run.program.version == "2018.6"

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == "MolecularDynamics"
    sec_method = sec_workflow.method
    assert sec_method.thermodynamic_ensemble == "NPT"
    assert sec_method.integrator_type == "leap_frog"
    assert sec_method.thermostat_parameters[0].thermostat_type == "nose_hoover"
    assert sec_method.thermostat_parameters[0].reference_temperature.magnitude == 298.0
    assert sec_method.thermostat_parameters[0].coupling_constant.magnitude == 5e-13
    assert sec_method.barostat_parameters[0].barostat_type == "parrinello_rahman"
    assert sec_method.barostat_parameters[0].coupling_type == "isotropic"
    assert np.all(
        sec_method.barostat_parameters[0].reference_pressure.magnitude
        == [[100000.0, 0.0, 0.0], [0.0, 100000.0, 0.0], [0.0, 0.0, 100000.0]]
    )
    assert np.all(
        sec_method.barostat_parameters[0].coupling_constant.magnitude
        == [
            [5.0e-12, 5.0e-12, 5.0e-12],
            [5.0e-12, 5.0e-12, 5.0e-12],
            [5.0e-12, 5.0e-12, 5.0e-12],
        ]
    )
    assert np.all(
        sec_method.barostat_parameters[0].compressibility.magnitude
        == [
            [7.4e-10, 0.0e00, 0.0e00],
            [0.0e00, 7.4e-10, 0.0e00],
            [0.0e00, 0.0e00, 7.4e-10],
        ]
    )
