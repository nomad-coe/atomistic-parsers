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
from atomisticparsers.dftbplus import DFTBPlusParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return DFTBPlusParser()


def test_static(parser):
    archive = EntryArchive()
    parser.parse('tests/data/dftbplus/static/slurm-259896.out', archive, None)

    sec_run = archive.run
    assert sec_run[0].program.version == 'development version (commit: 12acc2a)'
    assert sec_run[0].x_dftbp_parser_version == '8'

    sec_method = archive.run[0].method
    assert sec_method[0].tb.name == 'DFTB'
    assert (
        sec_method[0].tb.x_dftbp_input_parameters['Hamiltonian']['Mixer'][
            'InverseJacobiWeight'
        ]
        == 0.01
    )
    assert (
        sec_method[0].tb.x_dftbp_input_parameters['Analysis']['ElectronDynamics'][
            'Perturbation'
        ]['SpinType']
        == 'Singlet'
    )

    sec_system = archive.run[0].system
    assert len(sec_system[0].atoms.labels) == 1415
    assert sec_system[0].atoms.labels[10] == 'Ag'
    assert sec_system[0].atoms.positions[94][1].magnitude == approx(-1.52468396e-10)

    sec_scc = archive.run[0].calculation
    assert len(sec_scc) == 1
    assert sec_scc[0].energy.total.value.magnitude == approx(-1.97085965e-14)
    assert sec_scc[0].energy.total_t0.value.magnitude == approx(-1.97086582e-14)
    assert sec_scc[0].energy.x_dftbp_total_mermin.value.magnitude == approx(
        -1.970872e-14
    )
    assert len(sec_scc[0].scf_iteration) == 34
    assert sec_scc[0].scf_iteration[2].energy.total.value.magnitude == approx(
        -1.97066282e-14
    )
    assert sec_scc[0].scf_iteration[7].energy.change.magnitude == approx(-5.9194325e-19)


def test_relax(parser):
    archive = EntryArchive()
    parser.parse('tests/data/dftbplus/relax/output', archive, None)

    sec_system = archive.run[0].system
    assert len(sec_system) == 2
    assert len(sec_system[0].atoms.positions) == 192
    assert sec_system[0].atoms.positions[184][0].magnitude == approx(-1.57099201e-10)
    assert sec_system[0].atoms.labels[131] == 'C'
    assert sec_system[0].atoms.lattice_vectors[1][2].magnitude == approx(
        -2.30652463e-14
    )
    assert sec_system[1].atoms.positions[162][0].magnitude == approx(-4.85617227e-11)
    assert sec_system[1].atoms.labels[6] == 'O'
    assert sec_system[1].atoms.lattice_vectors[0][2].magnitude == approx(
        -3.78253529e-14
    )

    sec_scc = archive.run[0].calculation
    assert len(sec_scc) == 186
    assert sec_scc[121].energy.total.value.magnitude == approx(-1.29309979e-15)
    assert sec_scc[41].energy.x_dftbp_total_mermin.value.magnitude == approx(
        -1.2930996e-15
    )
    assert sec_scc[144].scf_iteration[2].energy.total.value.magnitude == approx(
        -1.3802862e-15
    )
    assert sec_scc[50].scf_iteration[4].energy.change.magnitude == approx(
        -3.79664524e-28
    )
    assert sec_scc[85].pressure.magnitude == approx(-13189912.1)
    assert sec_scc[-1].energy.fermi.magnitude == approx(-5.10615789e-19)
    assert sec_scc[-1].energy.nuclear_repulsion.value.magnitude == approx(
        8.30809449e-17
    )
    assert sec_scc[-1].energy.x_dftbp_band_t0.value.magnitude == approx(-1.3678808e-15)
    assert sec_scc[-1].energy.x_dftbp_band_free.value.magnitude == approx(
        -1.3678808e-15
    )
    assert sec_scc[-1].pressure.magnitude == approx(-324210.767)
    assert np.shape(sec_scc[-1].eigenvalues[0].energies) == (1, 184, 624)
    assert sec_scc[-1].eigenvalues[0].energies[0][172][54].magnitude == approx(
        -2.83213377e-18
    )
