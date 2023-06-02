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
from atomisticparsers.hoomdblue import HoomdblueParser
from nomad.datamodel.metainfo.simulation.workflow import MolecularDynamics


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return HoomdblueParser()


def test_md_basics(parser):
    archive = EntryArchive()
    parser.parse('tests/data/hoomdblue/gsd_one_frame/trajectory.gsd', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.name == 'HOOMDBLUE'

    sec_method = sec_run.method[0]
    sec_model = sec_method.force_field.model[0]
    angles = sec_model.contributions[1]
    assert angles.type == 'angles'
    assert angles.n_inter == 14800
    assert angles.n_atoms == 3
    assert angles.contributions[2].name == 'MartiniAngle-1'
    assert angles.contributions[2].n_inter == 3600
    assert angles.contributions[2].n_atoms == 3
    assert angles.contributions[2].atom_indices[-1][-1] == 68480
    assert angles.contributions[2].atom_labels[-1][-1] == 'C1'
    sec_atom_parameters = sec_method.atom_parameters[10]
    assert sec_atom_parameters.label == 'P4'
    assert sec_atom_parameters.mass.magnitude == approx(72)
    assert sec_atom_parameters.charge.magnitude == approx(0)

    sec_system = sec_run.system[0]
    sec_atoms = sec_system.atoms
    assert np.shape(sec_atoms.positions) == (68481, 3)
    assert sec_atoms.positions[2024][2].magnitude == approx(-3.4790456295013428)
    assert np.shape(sec_atoms.velocities) == (68481, 3)
    assert sec_atoms.velocities[4532][0].magnitude == approx(-0.05720171704888344)
    assert sec_atoms.lattice_vectors[2][2].magnitude == approx(13.735918998718262)
    sec_atomsgroup = sec_system.atoms_group
    assert len(sec_atomsgroup) == 17
    sec_molgroup = sec_atomsgroup[5]
    assert sec_molgroup.label == 'group_5'
    assert sec_molgroup.type == 'molecule_group'
    assert sec_molgroup.composition_formula == '5(50)'
    assert sec_molgroup.n_atoms == 650
    assert len(sec_molgroup.atom_indices) == 650
    assert sec_molgroup.atom_indices[100] == 47484
    assert sec_molgroup.is_molecule is False
    assert len(sec_molgroup.atoms_group) == 50
    sec_molecule = sec_molgroup.atoms_group[22]
    assert sec_molecule.label == '5'
    assert sec_molecule.type == 'molecule'
    assert sec_molecule.composition_formula == 'C1(2)C3(6)Na(2)Q0(1)Qa(1)ghost(1)'
    assert sec_molecule.n_atoms == 13
    assert len(sec_molecule.atom_indices) == 13
    assert sec_molecule.atom_indices[3] == 53731
    assert sec_molecule.is_molecule is True

    sec_calculation = sec_run.calculation[0]
    assert sec_calculation.step == 71850000

    sec_workflow = archive.workflow2
    assert type(sec_workflow) == MolecularDynamics
