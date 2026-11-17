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

import time

import pytest
import numpy as np

from nomad.datamodel import EntryArchive
from atomisticparsers.lammps import LammpsParser
from atomisticparsers.lammps.parser import LogParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return LammpsParser()


def test_nvt(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/lammps/hexane_cyclohexane/log.hexane_cyclohexane_nvt', archive, None
    )

    sec_run = archive.run[0]
    assert sec_run.program.version == '14 May 2016'

    sec_workflow = archive.workflow2
    assert sec_workflow.m_def.name == 'MolecularDynamics'
    assert sec_workflow.method.thermodynamic_ensemble == 'NVT'
    assert sec_workflow.method.integrator_type == 'velocity_verlet'
    assert sec_workflow.method.integration_timestep.magnitude == 2.5e-16
    assert sec_workflow.method.integration_timestep.units == 'second'
    assert sec_workflow.method.n_steps == 80000
    assert sec_workflow.method.coordinate_save_frequency == 400
    assert sec_workflow.method.thermodynamics_save_frequency == 400
    assert sec_workflow.method.thermostat_parameters[0].thermostat_type == 'nose_hoover'
    assert (
        sec_workflow.method.thermostat_parameters[0].reference_temperature.magnitude
        == 300.0
    )
    assert (
        sec_workflow.method.thermostat_parameters[0].reference_temperature.units
        == 'kelvin'
    )
    assert (
        sec_workflow.method.thermostat_parameters[0].coupling_constant.magnitude
        == 2.5e-14
    )
    assert (
        sec_workflow.method.thermostat_parameters[0].coupling_constant.units == 'second'
    )

    sec_method = sec_run.method[0]
    # data file (`atom_style full`): atom 1 is C (type 1), atom 7 is H (type 2)
    assert sec_method.atom_parameters[0].mass.to('amu').magnitude == approx(12.011)
    assert sec_method.atom_parameters[6].mass.to('amu').magnitude == approx(1.008)
    assert sec_method.atom_parameters[0].charge.to(
        'elementary_charge'
    ).magnitude == approx(-0.12)
    assert sec_method.atom_parameters[6].charge.to(
        'elementary_charge'
    ).magnitude == approx(0.06)
    assert len(sec_method.force_field.model[0].contributions) == 3
    assert sec_method.force_field.model[0].contributions[1].type == 'bond'
    assert sec_method.force_field.model[0].contributions[1].n_interactions == 666
    assert sec_method.force_field.model[0].contributions[1].n_atoms == 2
    assert sec_method.force_field.model[0].contributions[1].atom_indices[100, 1] == 103
    assert sec_method.force_field.model[0].contributions[1].atom_labels[350, 0] == '1'
    assert (
        sec_method.force_field.force_calculations.coulomb_cutoff.magnitude
        == 1.2000000000000002e-08
    )
    assert sec_method.force_field.force_calculations.coulomb_cutoff.units == 'meter'
    assert (
        sec_method.force_field.force_calculations.neighbor_searching.neighbor_update_frequency
        == 10
    )

    sec_system = sec_run.system
    assert len(sec_system) == 201
    assert sec_system[5].atoms.lattice_vectors[1][1].magnitude == approx(2.24235e-09)
    assert False not in sec_system[0].atoms.periodic
    assert sec_system[80].atoms.labels[91:96] == ['H', 'H', 'H', 'C', 'C']
    assert sec_system[0].atoms.bond_list[200, 0] == 194

    sec_scc = sec_run.calculation
    assert len(sec_scc) == 201
    assert sec_scc[21].energy.current.value.magnitude == approx(8.86689197e-18)
    assert sec_scc[56].pressure.magnitude == approx(-77642135.4975)
    assert sec_scc[103].temperature.magnitude == 291.4591
    assert sec_scc[11].step == 4400
    assert len(sec_scc[1].energy.contributions) == 9
    assert sec_scc[112].energy.contributions[8].kind == 'kspace long range'
    assert sec_scc[96].energy.contributions[2].value.magnitude == approx(1.19666271e-18)
    assert sec_scc[47].energy.contributions[4].value.magnitude == approx(1.42166035e-18)
    assert sec_scc[75].time_physical.magnitude == approx(83.56332225)
    assert sec_scc[112].time_calculation.magnitude == approx(1.2351 / 400)

    assert (
        sec_run.x_lammps_section_control_parameters[0].x_lammps_inout_control_atomstyle
        == 'full'
    )


def test_thermo_format(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/lammps/1_methyl_naphthalene/log.1_methyl_naphthalene', archive, None
    )

    sec_sccs = archive.run[0].calculation
    assert len(sec_sccs) == 301
    assert sec_sccs[98].energy.total.value.magnitude == approx(1.45322428e-17)

    assert len(archive.run[0].system) == 4


def test_traj_xyz(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/lammps/methane_xyz/log.methane_nvt_traj_xyz_thermo_style_custom',
        archive,
        None,
    )

    sec_systems = archive.run[0].system
    assert len(sec_systems) == 201
    assert sec_systems[13].atoms.positions[7][0].magnitude == approx(-8.00436e-10)


def test_traj_dcd(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/lammps/methane_dcd/log.methane_nvt_traj_dcd_thermo_style_custom',
        archive,
        None,
    )

    assert len(archive.run[0].calculation) == 201
    sec_systems = archive.run[0].system
    assert np.shape(sec_systems[56].atoms.positions) == (320, 3)
    assert len(sec_systems[107].atoms.labels) == 320


def test_unwrapped_pos(parser):
    archive = EntryArchive()
    parser.parse('tests/data/lammps/1_xyz_files/log.lammps', archive, None)

    assert len(archive.run[0].calculation) == 101
    sec_systems = archive.run[0].system
    assert sec_systems[1].atoms.positions[452][2].magnitude == approx(
        5.99898
    )  # JFR - units are incorrect?!
    assert sec_systems[2].atoms.velocities[457][-2].magnitude == approx(
        -0.928553
    )  # JFR - velocities are not being read!!


# TODO Fix dealing with multiple output files with archive_to_universe function, then add back in this test
# def test_multiple_dump(parser):
#     archive = EntryArchive()
#     parser.parse('tests/data/lammps/2_xyz_files/log.lammps', archive, None)

#     sec_systems = archive.run[0].system
#     assert len(sec_systems) == 101
#     assert sec_systems[2].atoms.positions[468][0].magnitude == approx(3.00831)
#     assert sec_systems[-1].atoms.velocities[72][1].magnitude == approx(-4.61496)  # JFR - universe cannot be built without positions


def test_md_atomsgroup(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/lammps/polymer_melt/Emin/log.step4.0_minimization', archive, None
    )

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
    assert (
        sec_systems[0].atoms_group[0].atoms_group[52].composition_formula
        == '1(1)2(1)3(1)4(1)5(1)6(1)7(1)8(1)9(1)10(1)'
    )
    assert sec_systems[0].atoms_group[0].atoms_group[52].n_atoms == 72
    assert sec_systems[0].atoms_group[0].atoms_group[52].atom_indices[8] == 3752
    assert sec_systems[0].atoms_group[0].atoms_group[52].is_molecule is True

    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].label == 'group_8'
    )
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].type
        == 'monomer_group'
    )
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].index == 7
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].composition_formula
        == '8(1)'
    )
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].n_atoms == 7
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atom_indices[5]
        == 5527
    )
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].is_molecule
        is False
    )

    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[7]
        .atoms_group[0]
        .label
        == '8'
    )
    assert (
        sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atoms_group[0].type
        == 'monomer'
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[7]
        .atoms_group[0]
        .index
        == 0
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[7]
        .atoms_group[0]
        .composition_formula
        == '1(4)4(2)6(1)'
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[7]
        .atoms_group[0]
        .n_atoms
        == 7
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[7]
        .atoms_group[0]
        .atom_indices[5]
        == 5527
    )
    assert (
        sec_systems[0]
        .atoms_group[0]
        .atoms_group[76]
        .atoms_group[7]
        .atoms_group[0]
        .is_molecule
        is False
    )


def test_geometry_optimization(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/lammps/polymer_melt/Emin/log.step4.0_minimization', archive, None
    )

    sec_workflow = archive.workflow2

    assert sec_workflow.method.type == 'atomic'
    assert sec_workflow.method.method == 'polak_ribiere_conjugant_gradient'

    assert (
        sec_workflow.method.convergence_tolerance_energy_difference.magnitude
        == approx(0.0)
    )
    assert sec_workflow.method.convergence_tolerance_energy_difference.units == 'joule'
    assert sec_workflow.results.final_energy_difference.magnitude == approx(0.0)
    assert sec_workflow.results.final_energy_difference.units == 'joule'

    assert sec_workflow.method.convergence_tolerance_force_maximum.magnitude == approx(
        100
    )
    assert sec_workflow.method.convergence_tolerance_force_maximum.units == 'newton'

    assert sec_workflow.results.final_force_maximum.magnitude == approx(5091750000.0)
    assert sec_workflow.results.final_force_maximum.units == 'newton'

    assert sec_workflow.method.optimization_steps_maximum == 10000
    assert sec_workflow.results.optimization_steps == 160
    assert len(sec_workflow.results.energies) == 159
    assert sec_workflow.results.energies[14].magnitude == approx(6.931486093999211e-17)
    assert sec_workflow.results.energies[14].units == 'joule'
    assert len(sec_workflow.results.steps) == 159
    assert sec_workflow.results.steps[22] == 1100


THERMO_HEADER = '   Step          Temp          Press     \n'
THERMO_LOOP = (
    'Loop time of 1.2 on 4 procs for 20 steps with 100 atoms\n\n'
    'Performance: 1.0 ns/day\n'
)


def thermo_rows(n_rows):
    return ''.join(
        f'{10 * i:9d} {300.0 + i:12.4f} {1.5:12.4f}\n' for i in range(n_rows)
    )


def thermo_frames(n_frames):
    return ''.join(
        f'---------------- Step {10 * i:8d} ----- CPU = {0.1 * i:10.4f} (sec) '
        '----------------\n'
        'TotEng   =     -4.6695 KinEng   =      2.2486 Temp     =      1.5000 \n'
        for i in range(n_frames)
    )


@pytest.mark.parametrize(
    'body, expected_steps',
    [
        (THERMO_HEADER + thermo_rows(3) + THERMO_LOOP, [0, 10, 20]),
        (
            THERMO_HEADER
            + thermo_rows(300)
            + 'ERROR: Lost atoms: original 100 current 99\n',
            None,
        ),
        (THERMO_HEADER + thermo_rows(300), None),
        (thermo_frames(3) + THERMO_LOOP, [0, 10, 20]),
        (thermo_frames(300), None),
        (
            THERMO_HEADER
            + thermo_rows(300)
            + 'ERROR: Lost atoms: original 100 current 99\n'
            + 'run 20\n'
            + THERMO_HEADER
            + thermo_rows(3)
            + THERMO_LOOP,
            [0, 10, 20],
        ),
        (
            THERMO_HEADER
            + thermo_rows(2)
            + THERMO_LOOP
            + 'run 1500000\n'
            + THERMO_HEADER
            + thermo_rows(300)
            + 'WARNING: Wall time limit reached (../timer.cpp:194)\n'
            + THERMO_LOOP,
            [0, 10],
        ),
        (
            thermo_frames(300)
            + 'ERROR: Lost atoms: original 100 current 99\n'
            + 'run 20\n'
            + thermo_frames(3)
            + THERMO_LOOP,
            [0, 10, 20],
        ),
        (
            THERMO_HEADER
            + thermo_rows(300)
            + 'run 20\n'
            + THERMO_HEADER
            + thermo_rows(3)
            + THERMO_LOOP,
            [0, 10, 20],
        ),
        (
            '\n' * 8000 + 'x y\n' + THERMO_HEADER + thermo_rows(3) + THERMO_LOOP,
            [0, 10, 20],
        ),
        (
            THERMO_HEADER
            + thermo_rows(5)
            + 'thermo_style multi\n'
            + 'run 20\n'
            + thermo_frames(3)
            + THERMO_LOOP,
            [0, 10, 20],
        ),
    ],
    ids=[
        'one-closed',
        'one-unterminated',
        'one-truncated-eof',
        'multi-closed',
        'multi-unterminated',
        'first-unclosed-then-closed',
        'warning-in-block',
        'multi-first-unclosed-then-closed',
        'unclosed-then-closed-no-separator',
        'blank-lines-before-block',
        'unclosed-column-then-closed-multi',
    ],
)
def test_thermo_block(tmp_path, body, expected_steps):
    """Only the first thermo block closed by `Loop time of` is used; unclosed blocks
    are skipped without catastrophic backtracking."""
    log_file = tmp_path / 'log.lammps'
    log_file.write_text('LAMMPS (29 Aug 2024)\nunits metal\nrun 20\n' + body)
    log_parser = LogParser()
    log_parser.mainfile = str(log_file)

    start = time.perf_counter()
    thermo_data = log_parser.get_thermodynamic_data()
    assert time.perf_counter() - start < 5

    if expected_steps is None:
        assert thermo_data is None
    else:
        assert [int(step) for step in thermo_data['Step']] == expected_steps


def test_dump_file_name_with_plus(tmp_path):
    """Command arguments are not cut off at characters such as `+`."""
    log_file = tmp_path / 'log.lammps'
    log_file.write_text(
        'LAMMPS (29 Aug 2024)\n'
        'dump di ion custom 100 ion_K+0.1_s1.dump id type xu yu zu\n'
    )
    log_parser = LogParser()
    log_parser.mainfile = str(log_file)

    assert log_parser.get_traj_files() == [str(tmp_path / 'ion_K+0.1_s1.dump')]


def test_read_data_cpu_echo(tmp_path):
    """The `read_data CPU = ...` echo triggers the directory scan for data files."""
    log_file = tmp_path / 'log.lammps'
    log_file.write_text(
        'LAMMPS (29 Aug 2024)\nread_data data.x\n  read_data CPU = 0.04 seconds\n'
    )
    (tmp_path / 'final.data').write_text('LAMMPS data file\n')
    log_parser = LogParser()
    log_parser.mainfile = str(log_file)

    assert log_parser.get_data_files() == [str(tmp_path / 'final.data')]


@pytest.mark.parametrize(
    'atom_style_line, atoms_header, expected_options',
    [
        ('atom_style atomic\n', 'Atoms\n', {'atom_style': 'id type x y z'}),
        (
            'atom_style full\n',
            'Atoms\n',
            {'atom_style': 'id resid type charge x y z'},
        ),
        (
            'atom_style ellipsoid\n',
            'Atoms\n',
            {'atom_style': 'id type ellipsoidflag density x y z'},
        ),
        (
            'atom_style hybrid full dipole\n',
            'Atoms\n',
            {'atom_style': 'id type x y z'},
        ),
        ('atom_style atomic\n', 'Atoms # full\n', {'atom_style': 'id type x y z'}),
        ('', 'Atoms # full\n', {'atom_style': 'id resid type charge x y z'}),
        ('', 'Atoms\n', {}),
        ('atom_style mystyle\n', 'Atoms\n', {}),
    ],
    ids=[
        'atom-style-atomic',
        'atom-style-full',
        'atom-style-ellipsoid',
        'atom-style-hybrid',
        'log-before-data-comment',
        'missing-with-data-comment',
        'missing-without-data-comment',
        'atom-style-unknown',
    ],
)
def test_mda_topology_options(
    tmp_path, atom_style_line, atoms_header, expected_options
):
    """The MDAnalysis column layout follows the log's `atom_style`, else the
    `Atoms # <style>` comment of the data file."""
    log_file = tmp_path / 'log.lammps'
    log_file.write_text('LAMMPS (29 Aug 2024)\nunits real\n' + atom_style_line)
    data_file = tmp_path / 'final.data'
    data_file.write_text(
        'LAMMPS data file\n\n1 atoms\n1 atom types\n\n'
        + atoms_header
        + '\n1 1 0.0 0.0 0.0\n'
    )
    lammps_parser = LammpsParser()
    lammps_parser.log_parser.mainfile = str(log_file)

    assert lammps_parser.get_mda_topology_options(str(data_file)) == expected_options
