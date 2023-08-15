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
from atomisticparsers.asap import AsapParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return AsapParser()


def test_geometry_optimization(parser):
    archive = EntryArchive()
    parser.parse('tests/data/asap/geo_opt1.traj', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '3.13.0b1'

    sec_method = sec_run.method[0]
    assert sec_method.force_field.model[0].name == 'emt'

    sec_sccs = sec_run.calculation
    assert sec_sccs[3].energy.total.value.magnitude == approx(7.51835442e-18)
    assert sec_sccs[10].forces.total.value[7][2].magnitude == approx(-3.72962848e-11)
    # assert sec_sccs[6].forces.total.value_raw[2][0].magnitude == approx(2.54691322e-10)

    sec_systems = sec_run.system
    assert sec_systems[4].atoms.positions[18][1].magnitude == approx(3.60873003e-10)
    assert sec_systems[9].atoms.lattice_vectors[0][0].magnitude == approx(1.083e-09)
    assert sec_systems[0].atoms.labels[11] == 'Cu'
    assert sec_systems[0].constraint[0].atom_indices == np.array(0)
    assert sec_systems[0].constraint[0].kind == 'fix_xy'


def test_molecular_dynamics(parser):
    archive = EntryArchive()
    parser.parse('tests/data/asap/moldyn1.traj', archive, None)

    sec_run = archive.run[0]

    sec_workfow = archive.workflow2

    assert sec_workfow.m_def.name == 'MolecularDynamics'
    assert sec_workfow.method.thermodynamic_ensemble == 'NVT'
    assert sec_workfow.x_asap_timestep == approx(0.4911347394232032)

    assert sec_run.system[8].atoms.velocities[11][2].magnitude == approx(-1291.224)
