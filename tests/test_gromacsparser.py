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

    sec_md = archive.workflow[0].molecular_dynamics
    assert sec_md.ensemble_type == 'NPT'
    assert sec_md.x_gromacs_integrator_dt.magnitude == 0.0005
    assert sec_md.x_gromacs_barostat_target_pressure.magnitude == approx(33333.33)

    sec_sccs = sec_run.calculation
    assert len(sec_sccs) == 7
    assert sec_sccs[2].energy.total.value.magnitude == approx(-3.2711290665182795e-17)
    assert sec_sccs[5].thermodynamics[0].pressure.magnitude == approx(-63926916.5)
    assert sec_sccs[-2].energy.contributions[1].value.magnitude == approx(-4.15778738e-17)
    assert sec_sccs[0].forces.total.value[5][2].magnitude == approx(-7.932968909721231e-10)

    sec_systems = sec_run.system
    assert len(sec_systems) == 2
    assert np.shape(sec_systems[0].atoms.positions) == (1516, 3)
    assert sec_systems[1].atoms.positions[800][1].magnitude == approx(2.4740036e-09)
    assert sec_systems[0].atoms.velocities[500][0].magnitude == approx(869.4773)
    assert sec_systems[1].atoms.lattice_vectors[2][2].magnitude == approx(2.469158e-09)

    sec_methods = sec_run.method
    assert len(sec_methods) == 1
    assert len(sec_methods[0].force_field.model[0].contributions) == 1127
    assert sec_methods[0].force_field.model[0].contributions[0].type == 'angle'
    assert sec_methods[0].force_field.model[0].contributions[1120].parameters[1] == 575.0


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

    assert sec_systems[0].atoms_group[0].label == 'seg_0_S1P1'
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

    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].label == 'ETHOX'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].type == 'monomer'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].index == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].composition_formula == 'C(2)H(4)O(1)'
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].n_atoms == 7
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].atom_indices[5] == 5527
    assert sec_systems[0].atoms_group[0].atoms_group[76].atoms_group[7].is_molecule is False


def test_RDF(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gromacs/fe_test/mdrun.out', archive, None)

    sec_workflow = archive.workflow[0]
    section_MD = sec_workflow.molecular_dynamics

    assert section_MD.ensemble_properties.label == 'molecular radial distribution functions'
    assert section_MD.ensemble_properties.n_smooth == 6

    assert section_MD.ensemble_properties.types[0] == 'SOL-Protein'
    assert section_MD.ensemble_properties.variables_name[1][0] == 'distance'
    assert section_MD.ensemble_properties.bins[0][0][122] == approx(10.330030603408813)
    assert section_MD.ensemble_properties.values[0][96] == approx(1.098907565374127)

    assert section_MD.ensemble_properties.types[1] == 'SOL-SOL'
    assert section_MD.ensemble_properties.variables_name[1][0] == 'distance'
    assert section_MD.ensemble_properties.bins[1][0][102] == approx(8.68381058692932)
    assert section_MD.ensemble_properties.values[1][55] == approx(1.0763463135639966)