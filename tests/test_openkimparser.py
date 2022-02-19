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
from atomisticparsers.openkim import OpenKIMParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return OpenKIMParser()


def test_entry(parser):
    archive = EntryArchive()
    parser.parse('tests/data/openkim/data.json', archive, None)

    sec_run = archive.run
    assert len(sec_run) == 10
    assert sec_run[1].program.version == 'TE_929921425793_007'

    sec_system = sec_run[3].system[0]
    assert sec_system.atoms.labels == ['Ag', 'Ag']
    assert sec_system.atoms.positions[1][1].magnitude == approx(1.65893189e-10)
    assert sec_system.atoms.lattice_vectors[2][2].magnitude == approx(3.31786378e-10)

    assert sec_run[0].method[0].force_field.model[0].name == 'LJ_ElliottAkerson_2015_Universal__MO_959249795837_003'

    sec_scc = sec_run[8].calculation[0]
    assert sec_scc.energy.total.value.magnitude == approx(4.513135831891813e-19)

    assert sec_run[0].x_openkim_meta['meta.runner.driver.name'] == 'LatticeConstantCubicEnergy'


def test_elastic(parser):
    archive = EntryArchive()
    parser.parse('tests/data/openkim/elastic-constants.json', archive, None)

    assert len(archive.run) == 3

    workflow = archive.workflow[0]
    assert workflow.type == 'elastic'
    assert workflow.elastic.elastic_constants_matrix_second_order[1][0].magnitude == approx(138253477422.5595)
    assert workflow.elastic.elastic_constants_gradient_matrix_second_order[15][16].magnitude == approx(7.155162012789679e-10)


def test_phonon(parser):
    archive = EntryArchive()
    parser.parse('tests/data/openkim/openkim_archive_phonon-dispersion-relation-cubic-crystal-npt.json', archive, None)

    sec_run = archive.run
    assert len(sec_run) == 337
    assert sec_run[0].calculation[0].stress.total.value[0][1].magnitude == approx(0)
    sec_band_structure = sec_run[15].calculation[0].band_structure_phonon[0]
    assert sec_band_structure.segment[0].energies[0][3][1].magnitude == approx(9.639834657408083e-22)
    assert np.shape(sec_band_structure.segment[0].kpoints) == (100, 3)

    workflow = archive.workflow
    assert len(workflow) == 337
    assert workflow[0].type == 'phonon'
    assert workflow[0].phonon.x_openkim_wave_number[0][8].magnitude == approx(2116441366.6289136)


def test_stacking_fault(parser):
    archive = EntryArchive()
    parser.parse('tests/data/openkim/StackingFaultFccCrystal_0bar_Ac__TE_567672586460_002.json', archive, None)

    sec_run = archive.run
    assert len(sec_run) == 6
    assert sec_run[1].system[0].atoms.labels == ['Ac']
    assert sec_run[0].system[0].atoms.positions[0][2].magnitude == approx(0.)
    assert sec_run[2].system[0].atoms.lattice_vectors[2][2].magnitude == approx(5.913618618249901e-10)

    workflow = archive.workflow
    assert len(workflow) == 6
    assert workflow[0].type == 'interface'
    assert workflow[0].interface.dimensionality == 2
    assert workflow[0].interface.shift_direction[1] == '110'
    assert workflow[0].interface.displacement_fraction[0][6] == approx(0.1224489795918367)
    assert np.shape(workflow[0].interface.gamma_surface) == (50, 50)
    assert workflow[0].interface.gamma_surface[0][16].magnitude == approx(-0.005698466524442978)
    assert workflow[1].interface.energy_unstable_stacking_fault.magnitude == approx(1.012205396160215)
    assert workflow[2].interface.energy_intrinsic_stacking_fault.magnitude == approx(-0.01268939047007063)
    assert workflow[3].interface.energy_unstable_twinning_fault.magnitude == approx(1.006221045274565)
    assert workflow[4].interface.energy_extrinsic_stacking_fault.magnitude == approx(-0.01357166427458997)
    assert workflow[5].interface.dimensionality == 1
    assert len(workflow[5].interface.displacement_fraction[0]) == 201
    assert workflow[5].interface.energy_fault_plane[18].magnitude == approx(0.3949196724956008)


def test_archive(parser):
    archive = EntryArchive()
    parser.parse('tests/data/openkim/openkim_archive_data.json', archive, None)

    assert len(archive.run) == 10
